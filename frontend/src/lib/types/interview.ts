export type InterviewType = 'full' | 'quick' | 'behavioral' | 'technical' | 'coding';
export type InterviewDifficulty = 'easy' | 'medium' | 'hard';
export type InterviewVoice = 'nova' | 'atlas' | 'aria';

export type InterviewState =
  | 'idle'
  | 'ai_speaking'
  | 'waiting_for_answer'
  | 'recording'
  | 'processing'
  | 'coding'
  | 'running_code'
  | 'reviewing_code'
  | 'completed'
  | 'ended_early';

export type AvatarState = 'idle' | 'speaking' | 'listening' | 'thinking';

export interface ResumeWorkExperience {
  id: string;
  company: string;
  role: string;
  duration: string;
  highlights: string[];
}

export interface ResumeEducation {
  id: string;
  school: string;
  degree: string;
  year: string;
}

export interface ResumeProject {
  id: string;
  name: string;
  description: string;
  tech: string[];
}

export interface CandidateProfile {
  name: string;
  title: string;
  summary: string;
  resumeFileName: string;
  lastUpdated: string;
  experience: ResumeWorkExperience[];
  education: ResumeEducation[];
  skills: string[];
  projects: ResumeProject[];
}

export interface ResumeSuggestion {
  id: string;
  category: 'impact' | 'phrasing' | 'tech' | 'action_verb' | 'keywords';
  title: string;
  currentText: string;
  suggestedText: string;
  explanation: string;
  status: 'pending' | 'applied' | 'dismissed';
}

export interface TargetJob {
  title: string;
  company: string;
  location: string;
  expectedDurationMinutes: number;
  questionCount: number;
  description: string;
  requiredSkills: string[];
  preferredSkills: string[];
}

// Match analysis is derived FROM a (candidate, job) pair — it is not part of
// the job itself. Kept separate so a real backend can later return
// "job" and "match analysis" as the distinct concepts they actually are,
// without reshaping TargetJob again.
export interface JobMatchAnalysis {
  score: number;
  matchedSkills: string[];
  partiallyMatchedSkills: string[];
  missingSkills: string[];
  focusAreas: string[];
}

// Input captured from the "Add/Replace Job Description" form, before mock
// analysis turns it into a full TargetJob.
export interface JobInput {
  title: string;
  company: string;
  location: string;
  description: string;
}

// Phase 2 workflow readiness for the Resume & Job page. Distinct from the
// Phase 1 `hasResumeContext`/`hasTargetJobContext`/`hasInterviewContext`
// booleans exposed by InterviewContext (those are the coarse gate other
// pages/guards read); these are the finer-grained status each mock stage
// moves through so the UI can render the right empty/loading/error/ready
// state without inferring anything from whether seed data happens to be
// non-empty.
export type ResumeStatus = 'empty' | 'parsing' | 'ready' | 'error';
export type JobStatus = 'empty' | 'analyzing' | 'ready' | 'error';
export type MatchStatus = 'idle' | 'analyzing' | 'ready' | 'error';

export type ResumeParseStage = 'uploading' | 'reading' | 'extracting';
export type JobAnalysisStage = 'analyzing' | 'extracting_required' | 'extracting_preferred';

// What resume parsing produces beyond the raw profile — enough to drive an
// interview on its own, with no Job Description required. A Target Job (when
// present) sharpens this via JobMatchAnalysis; it never replaces it.
export interface ResumeAnalysis {
  inferredRole: string;
  inferredSeniority?: string;
  coreSkills: string[];
  inferredFocusAreas: string[];
}

// Result of a resume upload — the parsed profile plus what could be inferred
// from it, so the mock service boundary already returns both pieces a real
// `POST /resume/parse` would.
export interface ParsedResume {
  candidate: CandidateProfile;
  analysis: ResumeAnalysis;
}

// Which inputs actually shaped the interview context for this session —
// lets /setup (and later, real question generation) distinguish a
// resume-only session from one sharpened by a target job. Not yet consumed
// for question generation; this phase only makes it available.
export type InterviewContextSource = 'resume' | 'resume_and_job';

export interface CodingTestCase {
  id: number;
  input: string;
  expected: string;
  hidden?: boolean;
}

export interface CodingTask {
  problemTitle: string;
  problemDescription: string;
  requirements: string[];
  constraints: string[];
  examples: Array<{
    input: string;
    output: string;
    explanation?: string;
  }>;
  starterCode: {
    [key: string]: string;
  };
  testCases: CodingTestCase[];
  solutionHint?: string;
}

export interface QuestionEvaluation {
  score: number;
  strengths: string[];
  improvements: string[];
  starFeedback?: {
    situation: string;
    task: string;
    action: string;
    result: string;
  };
  codingEvaluation?: {
    timeComplexity: string;
    spaceComplexity: string;
    cleanliness: string;
    edgeCasesHandled: boolean;
  };
}

export interface InterviewQuestion {
  id: string;
  index: number;
  total: number;
  type: 'behavioral' | 'technical' | 'system_design' | 'coding';
  questionText: string;
  followUpQuestion?: string;
  clarificationTips?: string[];
  candidateTranscript?: string;
  candidateAnswerTime?: number;
  codingTask?: CodingTask;
  submittedCode?: string;
  codeLanguage?: string;
  testCasesPassed?: number;
  totalTestCases?: number;
  evaluation?: QuestionEvaluation;
}

export interface CategoryScores {
  jobRelevance: number;
  technicalKnowledge: number;
  answerQuality: number;
  communication: number;
  problemSolving: number;
}

export interface SpeechMetrics {
  wordsPerMinute: number;
  longPauses: number;
  averageDurationSeconds: number;
  starAdherenceScore: number;
}

export interface InterviewReport {
  id: string;
  role: string;
  company: string;
  date: string;
  durationSeconds: number;
  overallScore: number;
  verdict: 'Strong Hire' | 'Hire' | 'Borderline' | 'Needs Practice';
  categoryScores: CategoryScores;
  speechMetrics: SpeechMetrics;
  strongAreas: string[];
  areasToImprove: string[];
  recommendedPractice: string[];
  questions: InterviewQuestion[];
}

export interface TranscriptMessage {
  id: string;
  sender: 'ai' | 'candidate';
  timestamp: string;
  text: string;
  isFollowUp?: boolean;
}
