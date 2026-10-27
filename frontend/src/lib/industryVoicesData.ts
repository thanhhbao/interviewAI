import { IndustryVoice } from './types/industryVoice';

// Curated dataset for the "Voices shaping modern work" ticker
// (IndustryVoicesTicker.tsx). Sourced from
// docs/interviewai_industry_voices_data_and_claude_prompt.md — do not add,
// invent, or paraphrase entries here; `contentType: 'quote'` text must stay
// verbatim to its `sourceUrl`, and `contentType: 'summary'` text is our own
// paraphrase of the linked remarks, never rendered in quotation marks.
//
// None of these entries claim or imply that the listed company or person
// endorses, uses, partners with, or sponsors InterviewAI — they are
// editorial commentary on AI/work/talent, each traceable to its own
// first-party source.
export const industryVoices: IndustryVoice[] = [
  {
    id: 'nvidia',
    company: 'NVIDIA',
    companyLogo: '/companies/nvidia.svg',
    person: 'Jensen Huang',
    role: 'Founder & CEO',
    text: 'It is essential to learn how to use AI — how to direct it, manage it, guardrail it, evaluate it.',
    contentType: 'quote',
    sourceLabel: 'NVIDIA · Davos / World Economic Forum · 2026',
    sourceUrl: 'https://blogs.nvidia.com/blog/davos-wef-blackrock-ceo-larry-fink-jensen-huang/',
    sourceDate: '2026',
  },
  {
    id: 'microsoft',
    company: 'Microsoft',
    companyLogo: '/companies/microsoft.svg',
    person: 'Satya Nadella',
    role: 'Chairman & CEO',
    text: 'A growth mindset keeps organizations learning, while AI becomes part of everyday knowledge work.',
    contentType: 'summary',
    sourceLabel: 'Microsoft · Leading through the AI Revolution',
    sourceUrl: 'https://unlocked.microsoft.com/gates-ballmer-nadella/',
  },
  {
    id: 'openai',
    company: 'OpenAI',
    companyLogo: '/companies/openai.svg',
    person: 'Sam Altman',
    role: 'CEO',
    text: 'Experts will probably still be much better than novices, as long as they embrace the new tools.',
    contentType: 'quote',
    sourceLabel: 'OpenAI · Jobs in the Intelligence Age · 2025',
    sourceUrl:
      'https://cdn.openai.com/global-affairs/06025361-1ede-4402-97d2-daf1e5918b43/jobs-in-the-intelligence-age-sept-2025.pdf',
    sourceDate: '2025',
  },
  {
    id: 'meta',
    company: 'Meta',
    companyLogo: '/companies/meta.svg',
    person: 'Mark Zuckerberg',
    role: 'Founder & CEO',
    text: 'AI should put greater capability in people’s hands so they can direct it toward what they value.',
    contentType: 'summary',
    sourceLabel: 'Meta · Personal Superintelligence for Everyone · 2025',
    sourceUrl: 'https://about.fb.com/news/2025/07/personal-superintelligence-for-everyone/',
    sourceDate: '2025',
  },
  {
    id: 'shopify',
    company: 'Shopify',
    companyLogo: '/companies/shopify.svg',
    person: 'Tobi Lütke',
    role: 'Co-founder & CEO',
    text: 'Reflexive AI usage is now a baseline expectation at Shopify.',
    contentType: 'quote',
    sourceLabel: 'Shopify · AI memo · 2025',
    sourceUrl: 'https://www.shopify.com/news/unserious-exploration',
    sourceDate: '2025',
  },
  {
    id: 'google',
    company: 'Google',
    companyLogo: '/companies/google.svg',
    person: 'Sundar Pichai',
    role: 'CEO, Google & Alphabet',
    text: 'AI is a once in a lifetime technology.',
    contentType: 'quote',
    sourceLabel: 'Google · AI Action Summit · 2025',
    sourceUrl: 'https://blog.google/innovation-and-ai/products/sundar-pichai-ai-action-summit/',
    sourceDate: '2025',
  },
  {
    id: 'amazon',
    company: 'Amazon',
    companyLogo: '/companies/amazon.svg',
    person: 'Andy Jassy',
    role: 'President & CEO',
    text: 'Agents will be teammates that we can call on at various stages of our work.',
    contentType: 'quote',
    sourceLabel: 'Amazon · Some thoughts on Generative AI · 2025',
    sourceUrl: 'https://www.aboutamazon.com/news/company-news/amazon-ceo-andy-jassy-on-generative-ai',
    sourceDate: '2025',
  },
  {
    id: 'github',
    company: 'GitHub',
    companyLogo: '/companies/github.svg',
    person: 'Thomas Dohmke',
    role: 'CEO at time of statement',
    text: 'Every developer deserves to be made 10 times more productive.',
    contentType: 'quote',
    sourceLabel: 'GitHub · Web Summit Rio · 2023',
    sourceUrl:
      'https://github.blog/developer-skills/github/web-summit-rio-2023-building-an-app-in-18-minutes-with-github-copilot-x/',
    sourceDate: '2023',
  },
];
