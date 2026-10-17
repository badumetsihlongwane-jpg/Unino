import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const rootDir = process.cwd();
const distDir = join(rootDir, 'dist');

// Strictly whitelisted directories and files needed for production web and Capacitor assets
const includeDirs = ['css', 'js', 'data', 'worker'];
const includeFiles = [
  'index.html',
  'landlord.html',
  'favicon.png',
  'unino-icon.png',
  'unino-icon-light.png',
  'unino-icon-dark.png',
  'NWUCAMP.geojson',
  'CNAME'
];

rmSync(distDir, { recursive: true, force: true });
mkdirSync(distDir, { recursive: true });

for (const dir of includeDirs) {
  const sourcePath = join(rootDir, dir);
  if (!existsSync(sourcePath)) continue;
  cpSync(sourcePath, join(distDir, dir), { recursive: true });
}

for (const file of includeFiles) {
  const sourcePath = join(rootDir, file);
  if (!existsSync(sourcePath)) continue;
  cpSync(sourcePath, join(distDir, file));
}

console.log(`Successfully prepared clean web bundle in ${distDir}`);