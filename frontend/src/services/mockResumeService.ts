import { CandidateProfile, ResumeAnalysis, ParsedResume, ResumeParseStage } from '@/lib/types/interview';
import { initialCandidateProfile } from '@/lib/mockData';

// Mock resume parsing boundary. Nothing here reads actual file bytes — it
// simulates the async shape (upload -> read -> extract) a real
// `POST /resume/parse` will have, so the orchestration in InterviewContext
// and the UI in /resume-job don't need to change when this is swapped for
// a real API call later.

const SUPPORTED_EXTENSIONS = ['.pdf', '.docx'];

export function isSupportedResumeFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return SUPPORTED_EXTENSIONS.some((ext) => name.endsWith(ext));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function inferSeniority(title: string): string | undefined {
  const t = title.toLowerCase();
  if (t.includes('staff') || t.includes('principal')) return 'Staff / Principal';
  if (t.includes('senior') || t.includes('sr.')) return 'Senior';
  if (t.includes('lead')) return 'Lead';
  if (t.includes('junior') || t.includes('associate') || t.includes('entry')) return 'Junior';
  return 'Mid-level';
}

// Deliberately not NLP — a small keyword -> focus-area mapping, just enough
// to make the suggested interview focus feel derived from the actual skill
// list instead of being static copy.
const FOCUS_AREA_RULES: Array<{ keywords: string[]; focus: string }> = [
  { keywords: ['fastapi', 'django', 'flask', 'express', 'spring', 'node', 'rest', 'grpc', 'graphql'], focus: 'API Design' },
  { keywords: ['postgresql', 'mysql', 'redis', 'mongodb', 'database', 'sql'], focus: 'Database Performance' },
  { keywords: ['docker', 'kubernetes', 'microservices', 'distributed'], focus: 'Distributed Systems' },
  { keywords: ['aws', 'gcp', 'azure', 'terraform', 'ecs', 'eks'], focus: 'Production Scalability' },
  { keywords: ['python', 'java', 'go', 'typescript', 'javascript', 'c++', 'rust'], focus: 'Backend Architecture' },
  { keywords: ['react', 'vue', 'frontend', 'ui'], focus: 'Frontend Engineering' },
  { keywords: ['kafka', 'rabbitmq', 'event', 'pub/sub'], focus: 'Event-Driven Architecture' },
];

function inferFocusAreas(skills: string[]): string[] {
  const lowerSkills = skills.map((s) => s.toLowerCase());
  const focusAreas: string[] = [];
  for (const rule of FOCUS_AREA_RULES) {
    const matches = rule.keywords.some((kw) => lowerSkills.some((s) => s.includes(kw)));
    if (matches && !focusAreas.includes(rule.focus)) focusAreas.push(rule.focus);
    if (focusAreas.length >= 5) break;
  }
  return focusAreas.length ? focusAreas : ['General role-specific problem solving'];
}

function inferResumeAnalysis(candidate: CandidateProfile): ResumeAnalysis {
  const inferredRole = candidate.experience[0]?.role || candidate.title;
  return {
    inferredRole,
    inferredSeniority: inferSeniority(candidate.title),
    coreSkills: candidate.skills.slice(0, 5),
    inferredFocusAreas: inferFocusAreas(candidate.skills),
  };
}

export async function parseResume(
  file: File,
  onStageChange?: (stage: ResumeParseStage) => void
): Promise<ParsedResume> {
  onStageChange?.('uploading');
  await delay(500);
  onStageChange?.('reading');
  await delay(600);
  onStageChange?.('extracting');
  await delay(700);

  // Mock result payload — a later real parser returns this same shape from
  // an actual document. Only the filename/timestamp are derived from the
  // real File the user picked; the rest is representative seed content.
  const candidate: CandidateProfile = {
    ...initialCandidateProfile,
    resumeFileName: file.name,
    lastUpdated: 'Just now',
  };

  return { candidate, analysis: inferResumeAnalysis(candidate) };
}
