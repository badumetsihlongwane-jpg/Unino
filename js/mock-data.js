// ============================================
// UNINO - Mock Data for Frontend MVP
// ============================================

const MockData = {
  // Current logged-in user
  currentUser: {
    id: 'u1',
    username: 'alex_k',
    firstName: 'Alex',
    lastName: 'Kim',
    email: 'alex.kim@university.edu',
    major: 'Computer Science',
    graduationYear: 2027,
    bio: 'CS major | Coffee addict | Building cool stuff ☕️💻',
    profilePicture: null,
    privacyMode: 'online', // online, offline, study
    locationSharing: true,
    karmaPoints: 342,
    studyStreakDays: 7,
    theme: 'dark',
    avatar: 'AK'
  },

  // Other users
  users: [
    { id: 'u2', username: 'sarah_j', firstName: 'Sarah', lastName: 'Johnson', major: 'Biology', graduationYear: 2026, bio: 'Pre-med grind 🔬', privacyMode: 'online', avatar: 'SJ', karmaPoints: 520, studyStreakDays: 14, distance: 120, status: 'Free Now', location: 'Student Center', courses: ['BIO201', 'CHEM101'] },
    { id: 'u3', username: 'mike_t', firstName: 'Mike', lastName: 'Thompson', major: 'Business', graduationYear: 2027, bio: 'Entrepreneur in training 📈', privacyMode: 'study', avatar: 'MT', karmaPoints: 180, studyStreakDays: 3, distance: 50, status: 'Study Mode', location: 'Library - 3rd Floor', courses: ['BUS301', 'ECON201'] },
    { id: 'u4', username: 'priya_r', firstName: 'Priya', lastName: 'Rao', major: 'Computer Science', graduationYear: 2027, bio: 'Full-stack dev & design nerd 🎨', privacyMode: 'online', avatar: 'PR', karmaPoints: 410, studyStreakDays: 21, distance: 200, status: 'Looking for Study Buddy', location: 'Coffee Shop', courses: ['CS201', 'MATH301'] },
    { id: 'u5', username: 'james_w', firstName: 'James', lastName: 'Williams', major: 'Psychology', graduationYear: 2026, bio: 'Understanding minds, one class at a time 🧠', privacyMode: 'online', avatar: 'JW', karmaPoints: 290, studyStreakDays: 5, distance: 350, status: 'Free Now', location: 'Psych Building', courses: ['PSY301', 'PSY401'] },
    { id: 'u6', username: 'emma_l', firstName: 'Emma', lastName: 'Lee', major: 'Graphic Design', graduationYear: 2028, bio: 'Making things pretty ✨', privacyMode: 'online', avatar: 'EL', karmaPoints: 155, studyStreakDays: 2, distance: 80, status: 'Free Now', location: 'Art Building', courses: ['ART101', 'DES201'] },
    { id: 'u7', username: 'david_c', firstName: 'David', lastName: 'Chen', major: 'Mathematics', graduationYear: 2026, bio: 'Numbers are beautiful 🔢', privacyMode: 'online', avatar: 'DC', karmaPoints: 670, studyStreakDays: 30, distance: 150, status: 'Study Mode', location: 'Math Lab', courses: ['MATH301', 'MATH401', 'CS201'] },
    { id: 'u8', username: 'nina_p', firstName: 'Nina', lastName: 'Patel', major: 'Engineering', graduationYear: 2027, bio: 'Building the future 🏗️', privacyMode: 'online', avatar: 'NP', karmaPoints: 380, studyStreakDays: 10, distance: 90, status: 'Looking for Study Buddy', location: 'Engineering Lab', courses: ['ENG201', 'MATH301'] },
    { id: 'u9', username: 'tyler_b', firstName: 'Tyler', lastName: 'Brooks', major: 'Film Studies', graduationYear: 2028, bio: 'Director in the making 🎬', privacyMode: 'offline', avatar: 'TB', karmaPoints: 90, studyStreakDays: 0, distance: 500, status: 'Offline', location: '', courses: ['FILM101', 'ART101'] },
    { id: 'u10', username: 'zoe_m', firstName: 'Zoe', lastName: 'Martinez', major: 'Chemistry', graduationYear: 2026, bio: 'Lab rat 🧪', privacyMode: 'online', avatar: 'ZM', karmaPoints: 445, studyStreakDays: 8, distance: 175, status: 'Free Now', location: 'Chemistry Lab', courses: ['CHEM101', 'CHEM301', 'BIO201'] },
  ],

  // Courses
  courses: [
    { id: 'c1', code: 'CS201', name: 'Data Structures & Algorithms', instructor: 'Prof. Williams', semester: 'Spring 2026', members: 45, unread: 3 },
    { id: 'c2', code: 'MATH301', name: 'Linear Algebra', instructor: 'Prof. Chen', semester: 'Spring 2026', members: 32, unread: 1 },
    { id: 'c3', code: 'CS301', name: 'Operating Systems', instructor: 'Prof. Garcia', semester: 'Spring 2026', members: 38, unread: 0 },
    { id: 'c4', code: 'PHIL101', name: 'Intro to Philosophy', instructor: 'Prof. Adams', semester: 'Spring 2026', members: 60, unread: 5 },
    { id: 'c5', code: 'ENG201', name: 'Technical Writing', instructor: 'Prof. Miller', semester: 'Spring 2026', members: 28, unread: 0 },
  ],

  // Assignment Circles
  assignmentCircles: [
    { id: 'ac1', courseCode: 'CS201', name: 'Assignment 3 - Binary Trees', members: ['u1', 'u4', 'u7'], dueDate: '2026-02-15', messages: 12 },
    { id: 'ac2', courseCode: 'MATH301', name: 'Problem Set 5', members: ['u1', 'u7', 'u8'], dueDate: '2026-02-12', messages: 8 },
    { id: 'ac3', courseCode: 'CS301', name: 'OS Lab Report 2', members: ['u1', 'u4'], dueDate: '2026-02-20', messages: 3 },
  ],

  // Marketplace Listings
  listings: [
    { id: 'l1', sellerId: 'u2', title: 'Organic Chemistry Textbook (8th Ed.)', description: 'Used for one semester, minor highlighting. Great condition overall.', category: 'textbook', price: 45.00, images: [], status: 'active', views: 23, createdAt: '2026-02-05', sellerName: 'Sarah J.', sellerAvatar: 'SJ', sellerRating: 4.8 },
    { id: 'l2', sellerId: 'u6', title: 'Logo Design & Branding Package', description: 'Custom logo, brand colors, typography. 3 concepts + revisions. Quick turnaround!', category: 'service', price: 75.00, images: [], status: 'active', views: 45, createdAt: '2026-02-03', sellerName: 'Emma L.', sellerAvatar: 'EL', sellerRating: 5.0 },
    { id: 'l3', sellerId: 'u3', title: 'Standing Desk (Adjustable)', description: 'Electric standing desk. Bought last year, moving out so need to sell.', category: 'furniture', price: 120.00, images: [], status: 'active', views: 67, createdAt: '2026-02-01', sellerName: 'Mike T.', sellerAvatar: 'MT', sellerRating: 4.5 },
    { id: 'l4', sellerId: 'u4', title: 'Web Development Tutoring', description: 'Learn React, Node.js, and full-stack dev. $25/hr. Flexible schedule.', category: 'service', price: 25.00, images: [], status: 'active', views: 89, createdAt: '2026-01-28', sellerName: 'Priya R.', sellerAvatar: 'PR', sellerRating: 4.9 },
    { id: 'l5', sellerId: 'u5', title: 'Psychology Study Guide Bundle', description: 'Complete study guides for PSY301 and PSY401. Notes + practice questions.', category: 'textbook', price: 15.00, images: [], status: 'active', views: 34, createdAt: '2026-02-07', sellerName: 'James W.', sellerAvatar: 'JW', sellerRating: 4.7 },
    { id: 'l6', sellerId: 'u7', title: 'Math Tutoring - All Levels', description: 'Calculus, Linear Algebra, Discrete Math. Patient and thorough. 30-day streak tutor!', category: 'service', price: 30.00, images: [], status: 'active', views: 112, createdAt: '2026-01-20', sellerName: 'David C.', sellerAvatar: 'DC', sellerRating: 5.0 },
    { id: 'l7', sellerId: 'u8', title: 'TI-84 Plus Calculator', description: 'Barely used, comes with case. Perfect for math/engineering courses.', category: 'electronics', price: 55.00, images: [], status: 'active', views: 41, createdAt: '2026-02-06', sellerName: 'Nina P.', sellerAvatar: 'NP', sellerRating: 4.6 },
    { id: 'l8', sellerId: 'u10', title: 'Lab Coat + Safety Goggles Set', description: 'Required for chem labs. Perfect condition. Selling because I have extras.', category: 'other', price: 20.00, images: [], status: 'active', views: 18, createdAt: '2026-02-08', sellerName: 'Zoe M.', sellerAvatar: 'ZM', sellerRating: 4.4 },
    { id: 'l9', sellerId: 'u9', title: 'Video Editing Services', description: 'Professional editing for projects, presentations, or social media. Final Cut Pro & Premiere.', category: 'service', price: 40.00, images: [], status: 'active', views: 56, createdAt: '2026-02-04', sellerName: 'Tyler B.', sellerAvatar: 'TB', sellerRating: 4.3 },
    { id: 'l10', sellerId: 'u6', title: 'Dorm Room Cleaning Service', description: 'Deep clean your dorm! $30 flat rate. Supplies included. Available weekends.', category: 'service', price: 30.00, images: [], status: 'active', views: 73, createdAt: '2026-02-02', sellerName: 'Emma L.', sellerAvatar: 'EL', sellerRating: 5.0 },
  ],

  // Events (Campus Pulse)
  events: [
    { id: 'e1', creatorId: 'u2', title: 'Study Session - Biology Midterm', type: 'study_session', location: 'Library Room 204', startTime: '2026-02-10T14:00', endTime: '2026-02-10T17:00', attendees: 12, maxAttendees: 20, description: 'Group study for BIO201 midterm. Bring your notes!', creatorName: 'Sarah J.', creatorAvatar: 'SJ' },
    { id: 'e2', creatorId: 'u3', title: 'Startup Pitch Night', type: 'meetup', location: 'Business School Auditorium', startTime: '2026-02-11T18:00', endTime: '2026-02-11T21:00', attendees: 45, maxAttendees: 100, description: 'Present your startup idea and get feedback from peers and professors.', creatorName: 'Mike T.', creatorAvatar: 'MT' },
    { id: 'e3', creatorId: 'u4', title: 'Hackathon Prep Meetup', type: 'study_session', location: 'CS Building Lab 3', startTime: '2026-02-12T10:00', endTime: '2026-02-12T15:00', attendees: 8, maxAttendees: 15, description: 'Planning session for the upcoming campus hackathon. All skill levels welcome!', creatorName: 'Priya R.', creatorAvatar: 'PR' },
    { id: 'e4', creatorId: 'u6', title: 'Art Show Opening Night', type: 'party', location: 'Student Gallery', startTime: '2026-02-14T19:00', endTime: '2026-02-14T22:00', attendees: 30, maxAttendees: 75, description: 'Valentine\'s Day art show! Student work, live music, and free food.', creatorName: 'Emma L.', creatorAvatar: 'EL' },
    { id: 'e5', creatorId: 'u5', title: 'Mental Health Walk & Talk', type: 'sports', location: 'Campus Quad', startTime: '2026-02-13T08:00', endTime: '2026-02-13T09:30', attendees: 20, maxAttendees: 50, description: 'Morning walk around campus. Great for de-stressing before midterms.', creatorName: 'James W.', creatorAvatar: 'JW' },
    { id: 'e6', creatorId: 'u1', title: 'Coffee & Code ☕', type: 'meetup', location: 'Campus Coffee Shop', startTime: '2026-02-10T09:00', endTime: '2026-02-10T11:00', attendees: 6, maxAttendees: 10, description: 'Casual coding session. Work on side projects, ask questions, drink coffee.', creatorName: 'Alex K.', creatorAvatar: 'AK' },
  ],

  // Messages / Conversations
  conversations: [
    {
      id: 'conv1', type: 'dm', name: 'Priya Rao', participantIds: ['u1', 'u4'], avatar: 'PR', lastMessage: 'Hey! Want to work on the CS201 assignment together?', lastMessageTime: '2026-02-09T10:30', unread: 2,
      messages: [
        { id: 'm1', senderId: 'u4', senderName: 'Priya', content: 'Hey Alex! Are you free today?', time: '2026-02-09T10:15' },
        { id: 'm2', senderId: 'u1', senderName: 'You', content: 'Hey! Yeah, I should be free after 2pm', time: '2026-02-09T10:20' },
        { id: 'm3', senderId: 'u4', senderName: 'Priya', content: 'Want to work on the CS201 assignment together?', time: '2026-02-09T10:25' },
        { id: 'm4', senderId: 'u4', senderName: 'Priya', content: 'I found a good approach for the binary tree problem', time: '2026-02-09T10:30' },
      ]
    },
    {
      id: 'conv2', type: 'dm', name: 'David Chen', participantIds: ['u1', 'u7'], avatar: 'DC', lastMessage: 'The proof for problem 3 uses induction', lastMessageTime: '2026-02-08T22:15', unread: 0,
      messages: [
        { id: 'm5', senderId: 'u7', senderName: 'David', content: 'Did you start the MATH301 problem set?', time: '2026-02-08T21:00' },
        { id: 'm6', senderId: 'u1', senderName: 'You', content: 'Just started, problem 3 is tricky', time: '2026-02-08T21:30' },
        { id: 'm7', senderId: 'u7', senderName: 'David', content: 'The proof for problem 3 uses induction', time: '2026-02-08T22:15' },
      ]
    },
    {
      id: 'conv3', type: 'dm', name: 'Sarah Johnson', participantIds: ['u1', 'u2'], avatar: 'SJ', lastMessage: 'Thanks for the help with the chem notes! 🙏', lastMessageTime: '2026-02-07T16:45', unread: 0,
      messages: [
        { id: 'm8', senderId: 'u2', senderName: 'Sarah', content: 'Do you have notes from Tuesday\'s chem lecture?', time: '2026-02-07T15:00' },
        { id: 'm9', senderId: 'u1', senderName: 'You', content: 'Yeah, let me share them with you', time: '2026-02-07T15:30' },
        { id: 'm10', senderId: 'u2', senderName: 'Sarah', content: 'Thanks for the help with the chem notes! 🙏', time: '2026-02-07T16:45' },
      ]
    },
    {
      id: 'conv4', type: 'course_room', name: 'CS201 - Data Structures', participantIds: ['u1', 'u4', 'u7'], avatar: 'CS', lastMessage: 'Midterm review session this Friday!', lastMessageTime: '2026-02-09T09:00', unread: 3,
      messages: [
        { id: 'm11', senderId: 'u7', senderName: 'David', content: 'Has anyone started the linked list implementation?', time: '2026-02-09T08:00' },
        { id: 'm12', senderId: 'u4', senderName: 'Priya', content: 'Yes! I can share my approach later today', time: '2026-02-09T08:30' },
        { id: 'm13', senderId: 'u7', senderName: 'David', content: 'Midterm review session this Friday!', time: '2026-02-09T09:00' },
      ]
    },
    {
      id: 'conv5', type: 'course_room', name: 'MATH301 - Linear Algebra', participantIds: ['u1', 'u7', 'u8'], avatar: 'MA', lastMessage: 'Prof posted extra practice problems', lastMessageTime: '2026-02-08T14:20', unread: 1,
      messages: [
        { id: 'm14', senderId: 'u8', senderName: 'Nina', content: 'Prof posted extra practice problems', time: '2026-02-08T14:20' },
      ]
    },
  ],

  // Connections / Friends
  connections: [
    { userId: 'u4', status: 'accepted' },
    { userId: 'u7', status: 'accepted' },
    { userId: 'u2', status: 'accepted' },
    { userId: 'u8', status: 'accepted' },
    { userId: 'u6', status: 'pending' },
    { userId: 'u5', status: 'pending' },
  ],

  // Achievements
  achievements: [
    { id: 'a1', name: 'First Steps', description: 'Complete your profile', icon: '👋', earned: true },
    { id: 'a2', name: 'Social Butterfly', description: 'Add 5 friends', icon: '🦋', earned: true },
    { id: 'a3', name: 'Study Streak 7', description: 'Study 7 days in a row', icon: '🔥', earned: true },
    { id: 'a4', name: 'Marketplace Maven', description: 'Make your first sale', icon: '💰', earned: false },
    { id: 'a5', name: 'Night Owl', description: 'Study after midnight', icon: '🦉', earned: true },
    { id: 'a6', name: 'Helper Hero', description: 'Earn 500 karma points', icon: '🦸', earned: false },
    { id: 'a7', name: 'Event Organizer', description: 'Create 3 events', icon: '🎉', earned: false },
    { id: 'a8', name: 'Study Streak 30', description: 'Study 30 days in a row', icon: '💎', earned: false },
  ],

  // Notifications
  notifications: [
    { id: 'n1', type: 'friend_request', from: 'Emma Lee', message: 'sent you a friend request', time: '5m ago', read: false },
    { id: 'n2', type: 'message', from: 'Priya Rao', message: 'sent you a message', time: '30m ago', read: false },
    { id: 'n3', type: 'event', from: 'Coffee & Code', message: 'starts in 1 hour', time: '1h ago', read: false },
    { id: 'n4', type: 'achievement', from: 'Study Streak 7', message: 'Achievement unlocked! 🔥', time: '2h ago', read: true },
    { id: 'n5', type: 'marketplace', from: 'Sarah J.', message: 'listed a new textbook', time: '3h ago', read: true },
  ]
};
