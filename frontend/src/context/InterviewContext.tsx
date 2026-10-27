'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, useRef, ReactNode } from 'react';
import {
  CandidateProfile,
  TargetJob,
  JobMatchAnalysis,
  ResumeAnalysis,
  JobInput,
  ResumeSuggestion,
  ResumeStatus,
  JobStatus,
  MatchStatus,
  ResumeParseStage,
  JobAnalysisStage,
  InterviewContextSource,
  InterviewType,
  InterviewDifficulty,
  InterviewVoice,
  InterviewState,
  AvatarState,
  InterviewQuestion,
  InterviewReport,
  TranscriptMessage,
} from '@/lib/types/interview';
import {
  initialCandidateProfile,
  initialTargetJob,
  initialJobMatchAnalysis,
  initialResumeAnalysis,
  initialResumeSuggestions,
  mockQuestions,
  sampleCompletedReport,
} from '@/lib/mockData';
import { runTwoSumCode, RunSuiteResult } from '@/lib/codeRunner';
import { parseResume, isSupportedResumeFile } from '@/services/mockResumeService';
import { analyzeJob } from '@/services/mockJobService';
import { analyzeMatch } from '@/services/mockMatchService';

interface InterviewContextType {
  // Data contexts
  candidate: CandidateProfile;
  setCandidate: React.Dispatch<React.SetStateAction<CandidateProfile>>;
  targetJob: TargetJob;
  setTargetJob: React.Dispatch<React.SetStateAction<TargetJob>>;
  suggestions: ResumeSuggestion[];
  applySuggestion: (id: string) => void;
  applyEditedSuggestion: (id: string, editedText: string) => void;
  dismissSuggestion: (id: string) => void;

  // Resume & Job workflow (Phase 2, corrected) — orchestrates the mock
  // services, keeps the parse/analyze/match stage machinery out of the UI
  // components. Resume is the only required input; Job/Match are an
  // optional enhancement layer — see `hasInterviewContext` below.
  resumeStatus: ResumeStatus;
  resumeError: string | null;
  resumeParseStage: ResumeParseStage | null;
  resumeAnalysis: ResumeAnalysis;
  uploadResume: (file: File) => Promise<void>;

  jobStatus: JobStatus;
  jobError: string | null;
  jobAnalysisStage: JobAnalysisStage | null;
  submitJob: (input: JobInput) => Promise<void>;
  clearTargetJob: () => void;

  matchAnalysis: JobMatchAnalysis;
  matchStatus: MatchStatus;
  matchError: string | null;
  retryMatchAnalysis: () => void;

  // Whether the current interview context was shaped by resume alone, or by
  // resume + a successfully analyzed target job. Not consumed for question
  // generation yet (that's a later phase) — just made available here.
  interviewContextSource: InterviewContextSource;

  updateCandidateProfile: (updated: CandidateProfile) => void;

  // Settings
  interviewType: InterviewType;
  setInterviewType: (type: InterviewType) => void;
  difficulty: InterviewDifficulty;
  setDifficulty: (diff: InterviewDifficulty) => void;
  voice: InterviewVoice;
  setVoice: (voice: InterviewVoice) => void;

  // Session state
  interviewState: InterviewState;
  avatarState: AvatarState;
  currentQuestionIndex: number;
  currentQuestion: InterviewQuestion;
  questions: InterviewQuestion[];
  isFollowUpActive: boolean;
  activeSubtitle: string;
  elapsedSeconds: number;
  answerTimerSeconds: number;
  isTranscriptOpen: boolean;
  setIsTranscriptOpen: (open: boolean) => void;
  isCameraActive: boolean;
  setIsCameraActive: (active: boolean) => void;
  transcript: TranscriptMessage[];

  // Coding workspace state
  activeCode: string;
  setActiveCode: (code: string) => void;
  activeLanguage: string;
  setActiveLanguage: (lang: string) => void;
  testResults: RunSuiteResult | null;
  isRunningCode: boolean;
  runCodeAction: () => void;
  submitCodingSolution: () => void;
  askForClarification: () => void;

  // Actions
  startInterviewSession: (type?: InterviewType) => void;
  startAnswer: () => void;
  finishAnswer: (transcriptText?: string) => void;
  restartAnswer: () => void;
  repeatCurrentQuestion: () => void;
  endInterviewSession: (status?: 'completed' | 'ended_early') => void;
  resetSession: () => void;
  latestReport: InterviewReport;

  // Workflow readiness — derived, not inferred from seed/mock data.
  // `hasInterviewContext` is the one route guards/CTAs actually gate on: the
  // Resume is the only REQUIRED piece — Job/Match are optional enhancement,
  // so this is true once the resume is parsed and neither job analysis nor
  // match analysis is still in flight (their success/failure doesn't matter,
  // only whether they've settled, so navigating away never leaves stale
  // "analyzing" state behind).
  hasResumeContext: boolean;
  hasTargetJobContext: boolean;
  hasInterviewContext: boolean;
  hasActiveSession: boolean;
  hasCompletedSession: boolean;
}

const InterviewContext = createContext<InterviewContextType | undefined>(undefined);

export const InterviewProvider = ({ children }: { children: ReactNode }) => {
  // Profiles & Jobs
  const [candidate, setCandidate] = useState<CandidateProfile>(initialCandidateProfile);
  const [targetJob, setTargetJob] = useState<TargetJob>(initialTargetJob);
  const [suggestions, setSuggestions] = useState<ResumeSuggestion[]>(initialResumeSuggestions);

  // Setup Options
  const [interviewType, setInterviewType] = useState<InterviewType>('full');
  const [difficulty, setDifficulty] = useState<InterviewDifficulty>('medium');
  const [voice, setVoice] = useState<InterviewVoice>('nova');

  // Interview Progression
  const [questions, setQuestions] = useState<InterviewQuestion[]>(mockQuestions);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState<number>(0);
  const [interviewState, setInterviewState] = useState<InterviewState>('idle');
  const [avatarState, setAvatarState] = useState<AvatarState>('idle');
  const [isFollowUpActive, setIsFollowUpActive] = useState<boolean>(false);
  const [activeSubtitle, setActiveSubtitle] = useState<string>('');
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [answerTimerSeconds, setAnswerTimerSeconds] = useState<number>(0);
  const [isTranscriptOpen, setIsTranscriptOpen] = useState<boolean>(false);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);

  // `hasPreparedSession` is true only for the lifetime of a session actually
  // started from Setup, so a fresh/direct visit to /interview (interviewState
  // still 'idle', or stale from history) can't pass the room's guard.
  const [hasPreparedSession, setHasPreparedSession] = useState<boolean>(false);
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);

  // Resume & Job workflow (Phase 2). Each starts 'empty'/'idle' even though
  // `candidate`/`targetJob` below are pre-seeded with demo data — that seed
  // data is for visual development elsewhere in the app (homepage, etc.),
  // not proof the user actually walked through Resume & Job. Route guards
  // and /resume-job's own UI only trust these statuses.
  const [resumeStatus, setResumeStatus] = useState<ResumeStatus>('empty');
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [resumeParseStage, setResumeParseStage] = useState<ResumeParseStage | null>(null);
  const [resumeAnalysis, setResumeAnalysis] = useState<ResumeAnalysis>(initialResumeAnalysis);

  const [jobStatus, setJobStatus] = useState<JobStatus>('empty');
  const [jobError, setJobError] = useState<string | null>(null);
  const [jobAnalysisStage, setJobAnalysisStage] = useState<JobAnalysisStage | null>(null);

  const [matchAnalysis, setMatchAnalysis] = useState<JobMatchAnalysis>(initialJobMatchAnalysis);
  const [matchStatus, setMatchStatus] = useState<MatchStatus>('idle');
  const [matchError, setMatchError] = useState<string | null>(null);

  // Coding State
  const [activeLanguage, setActiveLanguage] = useState<string>('python');
  const [activeCode, setActiveCode] = useState<string>(
    mockQuestions[3]?.codingTask?.starterCode?.python || ''
  );
  const [testResults, setTestResults] = useState<RunSuiteResult | null>(null);
  const [isRunningCode, setIsRunningCode] = useState<boolean>(false);

  // Report
  const [latestReport, setLatestReport] = useState<InterviewReport>(sampleCompletedReport);

  const currentQuestion = questions[currentQuestionIndex] || questions[0];

  // Dev visibility into what's actually driving the interview — currently
  // always the mock question set, but this reads `currentQuestion` itself
  // (not the mock array directly), so once a real AI response populates
  // `questions`/`currentQuestion` this keeps logging without any change.
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.log(
      `[Interview] Question ${currentQuestionIndex + 1}/${questions.length} (${currentQuestion.type}):`,
      currentQuestion
    );
  }, [currentQuestion, currentQuestionIndex, questions.length]);

  // Global Elapsed Timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (interviewState !== 'idle' && interviewState !== 'completed') {
      interval = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [interviewState]);

  // Candidate Answer Timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (interviewState === 'recording') {
      interval = setInterval(() => {
        setAnswerTimerSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      setAnswerTimerSeconds(0);
    }
    return () => clearInterval(interval);
  }, [interviewState]);

  // Runs mock match analysis for an explicit (candidate, job) pair — never
  // reads `candidate`/`targetJob` from closure, so callers that just built a
  // new candidate/job object (upload, edit, apply suggestion, job analysis)
  // can pass it straight in instead of racing a stale value from state that
  // hasn't re-rendered yet.
  const runMatchAnalysis = useCallback(async (candidateArg: CandidateProfile, jobArg: TargetJob) => {
    setMatchStatus('analyzing');
    setMatchError(null);
    try {
      const result = await analyzeMatch(candidateArg, jobArg);
      setMatchAnalysis(result);
      setMatchStatus('ready');
    } catch {
      setMatchStatus('error');
      setMatchError('Could not analyze your resume-job match. Please try again.');
    }
  }, []);

  const retryMatchAnalysis = useCallback(() => {
    if (resumeStatus === 'ready' && jobStatus === 'ready') {
      runMatchAnalysis(candidate, targetJob);
    }
  }, [resumeStatus, jobStatus, candidate, targetJob, runMatchAnalysis]);

  // Resume upload -> mock parse -> replace candidate profile. Any existing
  // match is explicitly invalidated first so a stale score never lingers
  // attached to the new resume while parsing/re-analysis is in flight.
  const uploadResume = useCallback(
    async (file: File) => {
      if (!isSupportedResumeFile(file)) {
        setResumeStatus('error');
        setResumeError('Unsupported file format. Please upload a PDF or DOCX resume.');
        return;
      }
      setResumeStatus('parsing');
      setResumeError(null);
      setResumeParseStage(null);
      if (jobStatus === 'ready') setMatchStatus('idle');
      try {
        const { candidate: parsedCandidate, analysis } = await parseResume(file, setResumeParseStage);
        setCandidate(parsedCandidate);
        setResumeAnalysis(analysis);
        setResumeStatus('ready');
        setResumeParseStage(null);
        if (jobStatus === 'ready') {
          await runMatchAnalysis(parsedCandidate, targetJob);
        }
      } catch {
        setResumeStatus('error');
        setResumeError('Something went wrong while parsing your resume. Please try again.');
        setResumeParseStage(null);
      }
    },
    [jobStatus, targetJob, runMatchAnalysis]
  );

  // Job description submit -> mock analysis -> replace target job. Same
  // invalidate-then-recalculate rule as resume upload.
  const submitJob = useCallback(
    async (input: JobInput) => {
      setJobStatus('analyzing');
      setJobError(null);
      setJobAnalysisStage(null);
      if (resumeStatus === 'ready') setMatchStatus('idle');
      try {
        const analyzed = await analyzeJob(input, setJobAnalysisStage);
        setTargetJob(analyzed);
        setJobStatus('ready');
        setJobAnalysisStage(null);
        if (resumeStatus === 'ready') {
          await runMatchAnalysis(candidate, analyzed);
        }
      } catch {
        setJobStatus('error');
        setJobError('Something went wrong while analyzing the job description. Please try again.');
        setJobAnalysisStage(null);
      }
    },
    [resumeStatus, candidate, runMatchAnalysis]
  );

  // Job is optional — removing it drops the workflow back to resume-only
  // (still fully valid) rather than leaving stale job/match data around
  // that no longer has a live source. The Resume itself is untouched.
  const clearTargetJob = useCallback(() => {
    setJobStatus('empty');
    setJobError(null);
    setJobAnalysisStage(null);
    setMatchStatus('idle');
    setMatchError(null);
  }, []);

  // Edit Profile save — full profile replacement (no re-upload required).
  const updateCandidateProfile = useCallback(
    (updated: CandidateProfile) => {
      setCandidate(updated);
      if (jobStatus === 'ready') {
        runMatchAnalysis(updated, targetJob);
      }
    },
    [jobStatus, targetJob, runMatchAnalysis]
  );

  // Resume suggestions handlers. Applying/editing a suggestion mutates the
  // candidate profile text and re-runs the real mock match service instead
  // of nudging a fixed score — the resulting number can still change by a
  // predictable amount, but it comes from actually re-evaluating skills.
  const applySuggestion = useCallback(
    (id: string) => {
      const target = suggestions.find((s) => s.id === id);
      if (!target) return;

      const updatedExperience = candidate.experience.map((exp) => ({
        ...exp,
        highlights: exp.highlights.map((h) =>
          h.includes(target.currentText) ? target.suggestedText : h
        ),
      }));
      const nextCandidate = { ...candidate, experience: updatedExperience };

      setCandidate(nextCandidate);
      setSuggestions((prev) => prev.map((s) => (s.id === id ? { ...s, status: 'applied' } : s)));
      if (jobStatus === 'ready') {
        runMatchAnalysis(nextCandidate, targetJob);
      }
    },
    [suggestions, candidate, jobStatus, targetJob, runMatchAnalysis]
  );

  // Same as applySuggestion, but the user-edited text (not the raw AI
  // suggestion) is what actually lands in the profile — the whole point of
  // Edit being a distinct action is that nothing gets applied unreviewed.
  const applyEditedSuggestion = useCallback(
    (id: string, editedText: string) => {
      const target = suggestions.find((s) => s.id === id);
      if (!target) return;

      const updatedExperience = candidate.experience.map((exp) => ({
        ...exp,
        highlights: exp.highlights.map((h) =>
          h.includes(target.currentText) ? editedText : h
        ),
      }));
      const nextCandidate = { ...candidate, experience: updatedExperience };

      setCandidate(nextCandidate);
      setSuggestions((prev) =>
        prev.map((s) => (s.id === id ? { ...s, status: 'applied', suggestedText: editedText } : s))
      );
      if (jobStatus === 'ready') {
        runMatchAnalysis(nextCandidate, targetJob);
      }
    },
    [suggestions, candidate, jobStatus, targetJob, runMatchAnalysis]
  );

  const dismissSuggestion = useCallback((id: string) => {
    setSuggestions((prev) =>
      prev.map((s) => (s.id === id ? { ...s, status: 'dismissed' } : s))
    );
  }, []);

  // Monotonic guard against a real race: cancelling an in-flight utterance
  // (speechSynthesis.cancel()) still fires that utterance's onend/onerror
  // asynchronously, which would otherwise invoke its stale onComplete after
  // a *newer* authoritative action (e.g. manually ending the interview)
  // already set the state it wanted — silently reverting it. Every call that
  // supersedes any pending speech bumps this ref; a callback only runs if
  // its own generation is still current.
  const speechGenerationRef = useRef(0);

  // AI Speech Simulation / Web Speech integration
  const speakQuestion = useCallback(
    (text: string, onComplete?: () => void) => {
      const generation = ++speechGenerationRef.current;
      setActiveSubtitle(text);
      setInterviewState('ai_speaking');
      setAvatarState('speaking');

      const runIfCurrent = () => {
        if (speechGenerationRef.current === generation && onComplete) onComplete();
      };

      // Try browser Web Speech API if supported
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;

        // Try to pick a pleasant voice
        const voices = window.speechSynthesis.getVoices();
        const preferredVoice = voices.find(
          (v) =>
            v.lang.startsWith('en') &&
            (v.name.includes('Samantha') ||
              v.name.includes('Daniel') ||
              v.name.includes('Google') ||
              v.name.includes('Natural'))
        );
        if (preferredVoice) utterance.voice = preferredVoice;

        utterance.onend = runIfCurrent;
        utterance.onerror = runIfCurrent;

        window.speechSynthesis.speak(utterance);
      } else {
        // Fallback timer simulation based on text length
        const duration = Math.min(8000, Math.max(3000, text.split(' ').length * 280));
        const timer = setTimeout(runIfCurrent, duration);
        return () => clearTimeout(timer);
      }
    },
    []
  );

  // Start interview session — the ONLY place that creates/resets a session.
  // Setup is the sole caller in the normal workflow; the Interview Room no
  // longer auto-starts itself, so `hasPreparedSession` here is what the
  // room's guard checks before rendering anything.
  const startInterviewSession = useCallback(
    (type?: InterviewType) => {
      if (type) setInterviewType(type);
      const startingIndex = type === 'coding' ? 3 : 0;
      setCurrentQuestionIndex(startingIndex);
      setElapsedSeconds(0);
      setIsFollowUpActive(false);
      setHasPreparedSession(true);

      // Clear stale state from any previous session/attempt so a new run
      // never inherits leftover test results or a mid-edit code draft.
      setTestResults(null);
      setActiveLanguage('python');
      setActiveCode(questions[3]?.codingTask?.starterCode?.python || '');

      const firstQ = questions[startingIndex];
      const candidateName = candidate.name || 'there';
      const companyName = targetJob.company ? ` with ${targetJob.company}` : '';
      const introText =
        startingIndex === 3
          ? firstQ.questionText
          : `Hello ${candidateName}, welcome to your interview${companyName}. ${firstQ.questionText}`;

      setTranscript([
        {
          id: `msg-${Date.now()}`,
          sender: 'ai',
          timestamp: '0:01',
          text: introText,
        },
      ]);

      speakQuestion(introText, () => {
        if (firstQ.type === 'coding') {
          setInterviewState('coding');
        } else {
          setInterviewState('waiting_for_answer');
        }
        setAvatarState('idle');
      });
    },
    [questions, speakQuestion, candidate, targetJob]
  );

  // Candidate initiates speaking
  const startAnswer = useCallback(() => {
    setInterviewState('recording');
    setAvatarState('listening');
  }, []);

  // Candidate finishes speaking
  const finishAnswer = useCallback(
    (customText?: string) => {
      const q = questions[currentQuestionIndex];
      const answerText = customText || q.candidateTranscript || 'Thank you. That covers my approach.';

      // Add to transcript
      const currentElapsed = `${Math.floor(elapsedSeconds / 60)}:${String(
        elapsedSeconds % 60
      ).padStart(2, '0')}`;

      setTranscript((prev) => [
        ...prev,
        {
          id: `msg-${Date.now()}`,
          sender: 'candidate',
          timestamp: currentElapsed,
          text: answerText,
        },
      ]);

      // Processing state
      setInterviewState('processing');
      setAvatarState('thinking');
      setActiveSubtitle('Understanding your answer...');

      setTimeout(() => {
        // Check if question has a follow-up and we haven't triggered it yet
        if (q.followUpQuestion && !isFollowUpActive) {
          setIsFollowUpActive(true);
          const followUpText = q.followUpQuestion;

          setTranscript((prev) => [
            ...prev,
            {
              id: `msg-${Date.now()}`,
              sender: 'ai',
              timestamp: currentElapsed,
              text: followUpText,
              isFollowUp: true,
            },
          ]);

          speakQuestion(followUpText, () => {
            setInterviewState('waiting_for_answer');
            setAvatarState('idle');
          });
        } else {
          // Move to next question or complete
          setIsFollowUpActive(false);
          const nextIndex = currentQuestionIndex + 1;

          if (nextIndex < questions.length) {
            setCurrentQuestionIndex(nextIndex);
            const nextQ = questions[nextIndex];

            // If next question is coding, transition
            if (nextQ.type === 'coding') {
              const codingIntro = nextQ.questionText;
              setTranscript((prev) => [
                ...prev,
                {
                  id: `msg-${Date.now()}`,
                  sender: 'ai',
                  timestamp: currentElapsed,
                  text: codingIntro,
                },
              ]);

              speakQuestion(codingIntro, () => {
                setInterviewState('coding');
                setAvatarState('idle');
              });
            } else {
              const nextPrompt = `Great point. Let’s move to the next topic. ${nextQ.questionText}`;
              setTranscript((prev) => [
                ...prev,
                {
                  id: `msg-${Date.now()}`,
                  sender: 'ai',
                  timestamp: currentElapsed,
                  text: nextPrompt,
                },
              ]);

              speakQuestion(nextPrompt, () => {
                setInterviewState('waiting_for_answer');
                setAvatarState('idle');
              });
            }
          } else {
            // Completion
            endInterviewSession();
          }
        }
      }, 2200);
    },
    [
      questions,
      currentQuestionIndex,
      elapsedSeconds,
      isFollowUpActive,
      speakQuestion,
    ]
  );

  // Restart answer
  const restartAnswer = useCallback(() => {
    setAnswerTimerSeconds(0);
  }, []);

  // Repeat question
  const repeatCurrentQuestion = useCallback(() => {
    const q = questions[currentQuestionIndex];
    const textToRepeat =
      isFollowUpActive && q.followUpQuestion ? q.followUpQuestion : q.questionText;
    speakQuestion(textToRepeat, () => {
      setInterviewState(q.type === 'coding' ? 'coding' : 'waiting_for_answer');
      setAvatarState('idle');
    });
  }, [questions, currentQuestionIndex, isFollowUpActive, speakQuestion]);

  // Code runner
  const runCodeAction = useCallback(() => {
    setIsRunningCode(true);
    setInterviewState('running_code');

    setTimeout(() => {
      const q = questions[currentQuestionIndex];
      const testCases = q.codingTask?.testCases || [];
      const results = runTwoSumCode(activeCode, activeLanguage, testCases);
      setTestResults(results);
      setIsRunningCode(false);
      setInterviewState('coding');
    }, 450);
  }, [questions, currentQuestionIndex, activeCode, activeLanguage]);

  // Submit coding solution
  const submitCodingSolution = useCallback(() => {
    setInterviewState('reviewing_code');
    setAvatarState('thinking');
    setActiveSubtitle('Analyzing algorithmic complexity and test suites...');

    setTimeout(() => {
      const feedback =
        'Excellent work on the Two Sum problem! You achieved an optimal O(n) runtime and O(n) memory footprint with a single-pass hash map, and passed all 4 test cases.';
      setTranscript((prev) => [
        ...prev,
        {
          id: `msg-${Date.now()}`,
          sender: 'ai',
          timestamp: `${Math.floor(elapsedSeconds / 60)}:${String(
            elapsedSeconds % 60
          ).padStart(2, '0')}`,
          text: feedback,
        },
      ]);

      speakQuestion(feedback, () => {
        endInterviewSession();
      });
    }, 2000);
  }, [elapsedSeconds, speakQuestion]);

  // Ask for clarification
  const askForClarification = useCallback(() => {
    const clarification =
      'Good question! You can assume each input has exactly one valid solution, and you should not use the same element twice. The output indices can be returned in any order.';
    speakQuestion(clarification, () => {
      setInterviewState('coding');
      setAvatarState('idle');
    });
  }, [speakQuestion]);

  // End Interview Session. `status` distinguishes natural completion (every
  // question answered — the internal callers below never pass an argument,
  // so they default to 'completed') from a manual early termination, which
  // the Interview Room passes explicitly only after the user confirms a
  // "you still have questions left" prompt. The two must not be conflated:
  // an early end shouldn't claim full completion, and shouldn't pad the
  // recorded duration up to a fake 15-minute floor.
  const endInterviewSession = useCallback(
    (status: 'completed' | 'ended_early' = 'completed') => {
      // Invalidate any speakQuestion callback still in flight (e.g. the
      // current question's utterance, cancelled below mid-speech) so it
      // can't fire after us and revert the state we're about to set.
      speechGenerationRef.current++;

      const closingText =
        status === 'completed'
          ? 'Great, that wraps up everything for today’s session. Your interview is complete, and your comprehensive evaluation report is ready for review.'
          : 'Your interview session has ended early. You can review a partial report or start a fresh practice session anytime.';
      setInterviewState(status);
      setAvatarState('speaking');
      setActiveSubtitle(closingText);

      // Save final report with actual elapsed duration. A full completion is
      // padded to a sane minimum since the mock report content assumes a
      // full session; an early end reports the real (shorter) time so it
      // doesn't misrepresent how far the candidate actually got.
      setLatestReport((prev) => ({
        ...prev,
        durationSeconds: status === 'completed' ? Math.max(elapsedSeconds, 900) : elapsedSeconds,
      }));

      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(closingText);
        window.speechSynthesis.speak(utterance);
      }
    },
    [elapsedSeconds]
  );

  // Reset Session
  const resetSession = useCallback(() => {
    setHasPreparedSession(false);
    setInterviewState('idle');
    setAvatarState('idle');
    setCurrentQuestionIndex(0);
    setElapsedSeconds(0);
    setIsFollowUpActive(false);
    setActiveSubtitle('');
    setTranscript([]);
    setTestResults(null);
  }, []);

  return (
    <InterviewContext.Provider
      value={{
        candidate,
        setCandidate,
        targetJob,
        setTargetJob,
        suggestions,
        applySuggestion,
        applyEditedSuggestion,
        dismissSuggestion,
        resumeStatus,
        resumeError,
        resumeParseStage,
        resumeAnalysis,
        uploadResume,
        jobStatus,
        jobError,
        jobAnalysisStage,
        submitJob,
        clearTargetJob,
        matchAnalysis,
        matchStatus,
        matchError,
        retryMatchAnalysis,
        interviewContextSource: jobStatus === 'ready' ? 'resume_and_job' : 'resume',
        updateCandidateProfile,
        interviewType,
        setInterviewType,
        difficulty,
        setDifficulty,
        voice,
        setVoice,
        interviewState,
        avatarState,
        currentQuestionIndex,
        currentQuestion,
        questions,
        isFollowUpActive,
        activeSubtitle,
        elapsedSeconds,
        answerTimerSeconds,
        isTranscriptOpen,
        setIsTranscriptOpen,
        isCameraActive,
        setIsCameraActive,
        transcript,
        activeCode,
        setActiveCode,
        activeLanguage,
        setActiveLanguage,
        testResults,
        isRunningCode,
        runCodeAction,
        submitCodingSolution,
        askForClarification,
        startInterviewSession,
        startAnswer,
        finishAnswer,
        restartAnswer,
        repeatCurrentQuestion,
        endInterviewSession,
        resetSession,
        latestReport,
        hasResumeContext: resumeStatus === 'ready',
        hasTargetJobContext: jobStatus === 'ready',
        // Resume is the only hard requirement. Job/Match are optional
        // enhancement — their success or failure doesn't gate anything,
        // only whether they're still actively running does, so the user
        // never navigates away mid-analysis into an inconsistent state.
        hasInterviewContext:
          resumeStatus === 'ready' && jobStatus !== 'analyzing' && matchStatus !== 'analyzing',
        hasActiveSession:
          hasPreparedSession && interviewState !== 'completed' && interviewState !== 'ended_early',
        hasCompletedSession:
          hasPreparedSession && (interviewState === 'completed' || interviewState === 'ended_early'),
      }}
    >
      {children}
    </InterviewContext.Provider>
  );
};

export const useInterview = () => {
  const context = useContext(InterviewContext);
  if (!context) {
    throw new Error('useInterview must be used within an InterviewProvider');
  }
  return context;
};
