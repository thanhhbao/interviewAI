// Generates the local monochrome company-logo assets referenced by
// `industryVoicesData.ts` (companyLogo: "/companies/<slug>.svg") from the
// `simple-icons` package — an open-source, CC0-licensed brand-icon set
// built exactly for embedding recognizable brand marks in third-party UI
// without hotlinking or redrawing logos by hand.
//
// Re-run with `npm run generate:company-logos` whenever a brand is added
// to industryVoicesData.ts, or if simple-icons later adds a brand this
// list currently lacks (see MISSING below).
//
// A logo is intentionally left out (and the ticker card falls back to a
// text-initial badge instead) rather than fabricated or hotlinked from
// elsewhere. As of simple-icons@16, these voices-ticker brands are not
// published there: Microsoft, OpenAI, Amazon.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ICONS_DIR = path.join(__dirname, '../node_modules/simple-icons/icons');
const OUT_DIR = path.join(__dirname, '../public/companies');

// Baked-in fill color rather than `currentColor`: these SVGs are loaded
// via <img src>, which renders in an isolated context and can't inherit
// `color` from the host page — so "monochrome" has to be a fixed value.
// Matches PALETTE.textPrimary in src/lib/motion.ts.
const MONO_COLOR = '#ECEDE9';

// outputName -> simple-icons slug. Keep in the same order as
// industryVoicesData.ts for easy comparison.
const LOGOS = {
  nvidia: 'nvidia',
  meta: 'meta',
  shopify: 'shopify',
  google: 'google',
  github: 'github',
};

// Present in industryVoicesData.ts but not (yet) in simple-icons — kept
// here so the gap is visible in source, not just in a chat transcript.
export const MISSING = ['microsoft', 'openai', 'amazon'];

mkdirSync(OUT_DIR, { recursive: true });

for (const [outName, slug] of Object.entries(LOGOS)) {
  const src = path.join(ICONS_DIR, `${slug}.svg`);
  const svg = readFileSync(src, 'utf8').replace('<svg ', `<svg fill="${MONO_COLOR}" `);
  writeFileSync(path.join(OUT_DIR, `${outName}.svg`), svg);
  console.log(`wrote public/companies/${outName}.svg`);
}

if (MISSING.length > 0) {
  console.log(`\nNo simple-icons entry for: ${MISSING.join(', ')} — falling back to text-initial badges.`);
}
