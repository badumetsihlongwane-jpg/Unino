/**
 * Unino Relaunch Date Shifter
 * 
 * Shifts old timestamps from launch (March/April 2026) to recent active timestamps (last 5-7 days up to today).
 * Preserves exact chronological order and relational integrity (users join before posting, comments follow posts).
 * 
 * Usage:
 *   node scripts/alter_launch_dates.cjs --dry-run
 *   node scripts/alter_launch_dates.cjs --execute
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const PROJECT_ID = 'unino-b215f';
const BASE_URL = `https://firestore.googleapis.com/v1/projects/${PROJECT_ID}/databases/(default)/documents`;
const BACKUP_PATH = path.resolve(__dirname, '../data/firestore_backup_pre_date_shift_2026-10-17.json');

const isExecute = process.argv.includes('--execute');
const isDryRun = !isExecute;

// Helper: HTTP request wrapper
function httpRequest(url, options = {}, body = null) {
  return new Promise((resolve, reject) => {
    const req = https.request(url, options, res => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const parsed = data ? JSON.parse(data) : {};
          resolve({ status: res.statusCode, data: parsed, raw: data });
        } catch (e) {
          resolve({ status: res.statusCode, raw: data, error: e.message });
        }
      });
    });
    req.on('error', reject);
    if (body) req.write(typeof body === 'string' ? body : JSON.stringify(body));
    req.end();
  });
}

// Fetch all docs from a collection
async function fetchCollection(collectionPath, pageSize = 100) {
  let docs = [];
  let pageToken = '';
  do {
    const url = `${BASE_URL}/${collectionPath}?pageSize=${pageSize}${pageToken ? `&pageToken=${pageToken}` : ''}`;
    const res = await httpRequest(url);
    if (res.status !== 200) {
      console.error(`Failed to fetch ${collectionPath}:`, res.status, res.raw);
      break;
    }
    const batch = res.data.documents || [];
    docs = docs.concat(batch);
    pageToken = res.data.nextPageToken || '';
  } while (pageToken);
  return docs;
}

// Update a single field on a document
async function patchDocumentField(docPath, fieldName, timestampIso) {
  const url = `${BASE_URL}/${docPath}?updateMask.fieldPaths=${fieldName}`;
  const payload = {
    fields: {
      [fieldName]: {
        timestampValue: timestampIso
      }
    }
  };
  return httpRequest(url, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(JSON.stringify(payload))
    }
  }, payload);
}

// Deterministic hash for reproducible pseudo-random offsets
function hashStr(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

async function main() {
  console.log(`\n════════════════════════════════════════════════════════════════`);
  console.log(` UNINO RELAUNCH DATE SHIFTER`);
  console.log(` Mode: ${isExecute ? '🔴 LIVE EXECUTION (WRITING TO FIRESTORE)' : '🟡 DRY RUN (PREVIEW ONLY)'}`);
  console.log(`════════════════════════════════════════════════════════════════\n`);

  console.log(`[1/5] Fetching existing documents from Firestore...`);
  const [rawPosts, rawUsers, rawAsg] = await Promise.all([
    fetchCollection('posts'),
    fetchCollection('users'),
    fetchCollection('assignmentGroups')
  ]);

  console.log(`  Fetched: ${rawPosts.length} posts, ${rawUsers.length} users, ${rawAsg.length} assignment groups.`);

  // Fetch comments for posts that have commentsCount > 0
  const postCommentsMap = {};
  let totalComments = 0;
  for (const p of rawPosts) {
    const pid = p.name.split('/').pop();
    const count = parseInt(p.fields?.commentsCount?.integerValue || '0', 10);
    if (count > 0) {
      const cDocs = await fetchCollection(`posts/${pid}/comments`);
      postCommentsMap[pid] = cDocs;
      totalComments += cDocs.length;
    }
  }
  console.log(`  Fetched: ${totalComments} comments across ${Object.keys(postCommentsMap).length} posts.`);

  // Save safety backup
  console.log(`\n[2/5] Creating safety backup to ${BACKUP_PATH}...`);
  const backupData = {
    backupDate: new Date().toISOString(),
    posts: rawPosts,
    postComments: postCommentsMap,
    users: rawUsers,
    assignmentGroups: rawAsg
  };
  fs.writeFileSync(BACKUP_PATH, JSON.stringify(backupData, null, 2), 'utf-8');
  console.log(`  Backup successfully written! (${(fs.statSync(BACKUP_PATH).size / 1024).toFixed(1)} KB)`);

  // Target timeline definition
  // NOW = Current execution time
  const now = Date.now();
  // Most recent post: ~25 minutes ago
  const endPostTime = now - (25 * 60 * 1000);
  // Oldest post: 6 days ago
  const startPostTime = now - (6 * 24 * 60 * 60 * 1000);
  const postSpan = endPostTime - startPostTime;

  console.log(`\n[3/5] Computing new timeline distribution...`);
  console.log(`  Reference NOW: ${new Date(now).toISOString()}`);
  console.log(`  Target Post Span: ${new Date(startPostTime).toISOString()} -> ${new Date(endPostTime).toISOString()}`);

  // Sort posts by original createdAt
  const sortedPosts = [...rawPosts].sort((a, b) => {
    const ta = new Date(a.fields?.createdAt?.timestampValue || 0).getTime();
    const tb = new Date(b.fields?.createdAt?.timestampValue || 0).getTime();
    return ta - tb;
  });

  const N = sortedPosts.length;
  const postUpdates = [];
  const authorEarliestActivity = {}; // uid -> min timestamp

  let lastTimestamp = startPostTime;

  for (let i = 0; i < N; i++) {
    const doc = sortedPosts[i];
    const pid = doc.name.split('/').pop();
    const authorId = doc.fields?.authorId?.stringValue || '';
    const origTsStr = doc.fields?.createdAt?.timestampValue || '';
    const authorName = doc.fields?.authorName?.stringValue || (doc.fields?.isAnonymous?.booleanValue ? 'Anonymous' : 'Unknown');

    // Use a slight power curve (1.25) so recent days get slightly higher density
    const progress = i / (N - 1);
    const curvedProgress = Math.pow(progress, 1.25);
    let targetTime = Math.round(startPostTime + curvedProgress * postSpan);

    // Jitter: add small pseudo-random variance based on post id
    const h = hashStr(pid);
    const jitterMin = ((h % 20) - 10) * 60 * 1000; // -10 to +10 mins
    targetTime += jitterMin;

    // Strict monotonicity: must be at least 12 minutes after previous post
    if (targetTime <= lastTimestamp + (12 * 60 * 1000)) {
      targetTime = lastTimestamp + (12 * 60 * 1000) + ((h % 10) * 60 * 1000);
    }
    // Cannot exceed endPostTime
    if (i === N - 1 || targetTime > endPostTime) {
      targetTime = endPostTime;
    }
    lastTimestamp = targetTime;

    const newTsIso = new Date(targetTime).toISOString();
    postUpdates.push({
      pid,
      docPath: `posts/${pid}`,
      origTs: origTsStr,
      newTs: newTsIso,
      authorId,
      authorName,
      text: (doc.fields?.text?.stringValue || '').slice(0, 35).replace(/\n/g, ' ')
    });

    if (authorId) {
      if (!authorEarliestActivity[authorId] || targetTime < authorEarliestActivity[authorId]) {
        authorEarliestActivity[authorId] = targetTime;
      }
    }
  }

  // Comments mapping
  const commentUpdates = [];
  for (const postUpdate of postUpdates) {
    const comments = postCommentsMap[postUpdate.pid] || [];
    if (!comments.length) continue;

    const postNewTime = new Date(postUpdate.newTs).getTime();
    const sortedComments = [...comments].sort((a, b) => {
      const ta = new Date(a.fields?.createdAt?.timestampValue || 0).getTime();
      const tb = new Date(b.fields?.createdAt?.timestampValue || 0).getTime();
      return ta - tb;
    });

    let lastCommentTime = postNewTime;
    for (let cIdx = 0; cIdx < sortedComments.length; cIdx++) {
      const cDoc = sortedComments[cIdx];
      const cid = cDoc.name.split('/').pop();
      const cAuthorId = cDoc.fields?.authorId?.stringValue || '';
      const cOrigTs = cDoc.fields?.createdAt?.timestampValue || '';
      const ch = hashStr(cid);

      // Comments spaced 5 to 25 mins after previous
      const gapMs = (5 + (ch % 20)) * 60 * 1000;
      let cNewTime = lastCommentTime + gapMs;
      if (cNewTime > now - (3 * 60 * 1000)) {
        cNewTime = now - (3 * 60 * 1000);
      }
      lastCommentTime = cNewTime;

      const cNewTsIso = new Date(cNewTime).toISOString();
      commentUpdates.push({
        pid: postUpdate.pid,
        cid,
        docPath: `posts/${postUpdate.pid}/comments/${cid}`,
        origTs: cOrigTs,
        newTs: cNewTsIso,
        authorId: cAuthorId
      });

      if (cAuthorId) {
        if (!authorEarliestActivity[cAuthorId] || cNewTime < authorEarliestActivity[cAuthorId]) {
          authorEarliestActivity[cAuthorId] = cNewTime;
        }
      }
    }
  }

  // Users mapping
  const userUpdates = [];
  const userEarliestJoin = now - (14 * 24 * 60 * 60 * 1000); // 14 days ago max
  for (const uDoc of rawUsers) {
    const uid = uDoc.name.split('/').pop();
    const origJoinedAt = uDoc.fields?.joinedAt?.timestampValue || '';
    const name = uDoc.fields?.displayName?.stringValue || uDoc.fields?.firstName?.stringValue || 'User';
    const uh = hashStr(uid);

    let newJoinTime;
    if (authorEarliestActivity[uid]) {
      // Author joined between 6h and 48h before their first post/comment
      const leadTimeMs = (6 + (uh % 42)) * 60 * 60 * 1000;
      newJoinTime = authorEarliestActivity[uid] - leadTimeMs;
    } else {
      // Non-author joined between 1 and 12 days ago
      const daysAgo = 1 + (uh % 11);
      const hoursAgo = (uh % 24);
      newJoinTime = now - (daysAgo * 24 * 60 * 60 * 1000) - (hoursAgo * 60 * 60 * 1000);
    }
    if (newJoinTime < userEarliestJoin) {
      newJoinTime = userEarliestJoin + ((uh % 48) * 60 * 60 * 1000);
    }

    const newJoinedIso = new Date(newJoinTime).toISOString();
    userUpdates.push({
      uid,
      docPath: `users/${uid}`,
      name,
      origJoinedAt,
      newJoinedAt: newJoinedIso
    });
  }

  // Print Preview Sample
  console.log(`\n[4/5] Timeline Preview (Sample):`);
  console.log(`\n  POSTS (Oldest 3 and Newest 5):`);
  const samplePosts = [...postUpdates.slice(0, 3), ...postUpdates.slice(-5)];
  samplePosts.forEach(p => {
    const hoursAgo = ((now - new Date(p.newTs).getTime()) / (3600 * 1000)).toFixed(1);
    console.log(`    ${p.origTs.slice(0, 10)} -> ${p.newTs} (${hoursAgo}h ago) | ${p.authorName.padEnd(16)} | "${p.text}"`);
  });

  console.log(`\n  COMMENTS Sample:`);
  commentUpdates.slice(0, 4).forEach(c => {
    const minsAgo = ((now - new Date(c.newTs).getTime()) / (60 * 1000)).toFixed(0);
    console.log(`    Post ${c.pid.slice(0, 8)}.. Comment ${c.cid.slice(0, 8)}.. -> ${c.newTs} (${minsAgo}m ago)`);
  });

  console.log(`\n  USERS Sample (Joined Dates):`);
  userUpdates.slice(0, 5).forEach(u => {
    const daysAgo = ((now - new Date(u.newJoinedAt).getTime()) / (24 * 3600 * 1000)).toFixed(1);
    console.log(`    ${u.name.padEnd(18)} | ${u.origJoinedAt.slice(0, 10)} -> ${u.newJoinedAt} (${daysAgo} days ago)`);
  });

  // Summary counts
  console.log(`\n  Total to update:`);
  console.log(`    - ${postUpdates.length} Posts (createdAt)`);
  console.log(`    - ${commentUpdates.length} Comments (createdAt)`);
  console.log(`    - ${userUpdates.length} Users (joinedAt)`);

  // Execution
  if (!isExecute) {
    console.log(`\n[5/5] DRY RUN COMPLETE. No data was modified.`);
    console.log(`  To apply these updates to Firestore, run:`);
    console.log(`    node scripts/alter_launch_dates.cjs --execute\n`);
    return;
  }

  console.log(`\n[5/5] EXECUTING LIVE UPDATES TO FIRESTORE...`);

  // Update Posts
  console.log(`  Updating ${postUpdates.length} posts...`);
  let postSuccess = 0;
  for (const p of postUpdates) {
    const res = await patchDocumentField(p.docPath, 'createdAt', p.newTs);
    if (res.status === 200) postSuccess++;
    else console.error(`    Failed post ${p.pid}:`, res.status, res.raw);
  }
  console.log(`  Posts completed: ${postSuccess}/${postUpdates.length}`);

  // Update Comments
  console.log(`  Updating ${commentUpdates.length} comments...`);
  let commentSuccess = 0;
  for (const c of commentUpdates) {
    const res = await patchDocumentField(c.docPath, 'createdAt', c.newTs);
    if (res.status === 200) commentSuccess++;
    else console.error(`    Failed comment ${c.cid}:`, res.status, res.raw);
  }
  console.log(`  Comments completed: ${commentSuccess}/${commentUpdates.length}`);

  // Update Users
  console.log(`  Updating ${userUpdates.length} users...`);
  let userSuccess = 0;
  for (const u of userUpdates) {
    const res = await patchDocumentField(u.docPath, 'joinedAt', u.newJoinedAt);
    if (res.status === 200) userSuccess++;
    else console.error(`    Failed user ${u.uid}:`, res.status, res.raw);
  }
  console.log(`  Users completed: ${userSuccess}/${userUpdates.length}`);

  console.log(`\n════════════════════════════════════════════════════════════════`);
  console.log(` ALL UPDATES COMPLETED SUCCESSFULLY!`);
  console.log(`════════════════════════════════════════════════════════════════\n`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
