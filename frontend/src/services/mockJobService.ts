import { TargetJob, JobInput, JobAnalysisStage } from '@/lib/types/interview';

// Mock job-description analysis boundary. `analyzeJob` stands in for a
// future `POST /job/analyze` — it takes the same JobInput shape a real
// endpoint would and returns a full TargetJob, so /resume-job never needs
// to change when the mock is swapped out.

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Deliberately not NLP — a small known-skill vocabulary the mock scans the
// pasted description for, just enough to make required/preferred feel
// derived from the actual text instead of static.
const KNOWN_SKILLS = [
  'Python', 'JavaScript', 'TypeScript', 'React', 'Node.js', 'FastAPI', 'Django',
  'PostgreSQL', 'MySQL', 'MongoDB', 'Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure',
  'Redis', 'Kafka', 'gRPC', 'REST', 'GraphQL', 'Microservices', 'CI/CD',
  'Go', 'Java', 'C++', 'Rust', 'Terraform', 'OpenTelemetry', 'Datadog',
];

function extractSkillsFromText(text: string): string[] {
  const lower = text.toLowerCase();
  return KNOWN_SKILLS.filter((skill) => lower.includes(skill.toLowerCase()));
}

export async function analyzeJob(
  input: JobInput,
  onStageChange?: (stage: JobAnalysisStage) => void
): Promise<TargetJob> {
  onStageChange?.('analyzing');
  await delay(600);

  onStageChange?.('extracting_required');
  await delay(600);
  const found = extractSkillsFromText(input.description);
  const requiredCount = Math.max(1, Math.ceil(found.length * 0.6));
  const requiredSkills = found.slice(0, requiredCount);

  onStageChange?.('extracting_preferred');
  await delay(500);
  const preferredSkills = found.slice(requiredCount);

  return {
    title: input.title,
    company: input.company || 'Target Company',
    location: input.location || 'Not specified',
    expectedDurationMinutes: 25,
    questionCount: 4,
    description: input.description,
    requiredSkills: requiredSkills.length ? requiredSkills : ['Core problem solving', 'Relevant domain experience'],
    preferredSkills: preferredSkills.length ? preferredSkills : ['Communication', 'Team collaboration'],
  };
}
