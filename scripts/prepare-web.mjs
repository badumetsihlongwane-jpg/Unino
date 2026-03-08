import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const rootDir = process.cwd();
const distDir = join(rootDir, 'dist');
const includePaths = ['index.html', 'css', 'js', 'worker', 'Start'];

rmSync(distDir, { recursive: true, force: true });
mkdirSync(distDir, { recursive: true });

for (const relativePath of includePaths) {
  const sourcePath = join(rootDir, relativePath);
  if (!existsSync(sourcePath)) continue;
  cpSync(sourcePath, join(distDir, relativePath), { recursive: true });
}

const topLevelFiles = readdirSync(rootDir, { withFileTypes: true })
  .filter(entry => entry.isFile())
  .map(entry => entry.name)
  .filter(name => !['package.json', 'package-lock.json', 'capacitor.config.ts'].includes(name));

for (const fileName of topLevelFiles) {
  cpSync(join(rootDir, fileName), join(distDir, fileName));
}

console.log(`Prepared web bundle in ${distDir}`);