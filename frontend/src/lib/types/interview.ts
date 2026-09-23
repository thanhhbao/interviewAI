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
  | 'completed';

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
  matchScore: number;
  matchedSkills: string[];
  missingSkills: string[];
  partiallyMatchedSkills: string[];
  focusAreas: string[];
}

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
