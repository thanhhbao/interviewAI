import { CandidateProfile, TargetJob, JobMatchAnalysis } from '@/lib/types/interview';

// Mock resume<->job match boundary, standing in for a future
// `POST /match/analyze`. Deterministic and based on the actual candidate
// skills vs. the actual job's required/preferred skills — not a fixed
// score — so it genuinely changes when either side changes.

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalize(skill: string): string {
  return skill.toLowerCase().replace(/[^a-z0-9]/g, '');
}

function isSkillPresent(candidateSkillsNorm: string[], skill: string): boolean {
  const n = normalize(skill);
  return candidateSkillsNorm.some((cs) => cs.length > 2 && (cs.includes(n) || n.includes(cs)));
}

export async function analyzeMatch(
  candidate: CandidateProfile,
  job: TargetJob
): Promise<JobMatchAnalysis> {
  await delay(900);

  const candidateSkillsNorm = candidate.skills.map(normalize);

  const matchedSkills: string[] = [];
  const missingSkills: string[] = [];
  job.requiredSkills.forEach((skill) => {
    if (isSkillPresent(candidateSkillsNorm, skill)) matchedSkills.push(skill);
    else missingSkills.push(skill);
  });

  const partiallyMatchedSkills: string[] = [];
  job.preferredSkills.forEach((skill) => {
    if (isSkillPresent(candidateSkillsNorm, skill)) matchedSkills.push(skill);
    else partiallyMatchedSkills.push(skill);
  });

  const totalConsidered = job.requiredSkills.length + job.preferredSkills.length || 1;
  const rawScore =
    ((matchedSkills.length + partiallyMatchedSkills.length * 0.5) / totalConsidered) * 100;
  const score = Math.max(0, Math.min(100, Math.round(rawScore)));

  const focusAreas = [...missingSkills, ...partiallyMatchedSkills]
    .slice(0, 4)
    .map((skill) => `Deepen familiarity with ${skill}`);

  return {
    score,
    matchedSkills,
    partiallyMatchedSkills,
    missingSkills,
    focusAreas: focusAreas.length ? focusAreas : ['General role-specific problem solving'],
  };
}
