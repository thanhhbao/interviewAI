'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import {
  CandidateProfile,
  TargetJob,
  ResumeSuggestion,
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
  initialResumeSuggestions,
  mockQuestions,
  sampleCompletedReport,
} from '@/lib/mockData';
import { runTwoSumCode, RunSuiteResult } from '@/lib/codeRunner';

interface InterviewContextType {
  // Data contexts
  candidate: CandidateProfile;
  setCandidate: React.Dispatch<React.SetStateAction<CandidateProfile>>;
  targetJob: TargetJob;
  setTargetJob: React.Dispatch<React.SetStateAction<TargetJob>>;
  suggestions: ResumeSuggestion[];
  applySuggestion: (id: string) => void;
  dismissSuggestion: (id: string) => void;

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
  endInterviewSession: () => void;
  resetSession: () => void;
  latestReport: InterviewReport;
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
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);

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

  // Resume suggestions handlers
  const applySuggestion = useCallback(
    (id: string) => {
      const target = suggestions.find((s) => s.id === id);
      if (!target) return;

      // Update candidate profile work experience text
      setCandidate((prev) => {
        const updatedExp = prev.experience.map((exp) => ({
          ...exp,
          highlights: exp.highlights.map((h) =>
            h.includes(target.currentText) ? target.suggestedText : h
          ),
        }));
        return {
          ...prev,
          experience: updatedExp,
        };
      });

      // Recalculate job match score dynamically (+4 points)
      setTargetJob((prev) => ({
        ...prev,
        matchScore: Math.min(96, prev.matchScore + 4),
      }));

      // Mark suggestion as applied
      setSuggestions((prev) =>
        prev.map((s) => (s.id === id ? { ...s, status: 'applied' } : s))
      );
    },
    [suggestions]
  );

  const dismissSuggestion = useCallback((id: string) => {
    setSuggestions((prev) =>
      prev.map((s) => (s.id === id ? { ...s, status: 'dismissed' } : s))
    );
  }, []);

  // AI Speech Simulation / Web Speech integration
  const speakQuestion = useCallback(
    (text: string, onComplete?: () => void) => {
      setActiveSubtitle(text);
      setInterviewState('ai_speaking');
      setAvatarState('speaking');

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

        utterance.onend = () => {
          if (onComplete) onComplete();
        };
        utterance.onerror = () => {
          if (onComplete) onComplete();
        };

        window.speechSynthesis.speak(utterance);
      } else {
        // Fallback timer simulation based on text length
        const duration = Math.min(8000, Math.max(3000, text.split(' ').length * 280));
        const timer = setTimeout(() => {
          if (onComplete) onComplete();
        }, duration);
        return () => clearTimeout(timer);
      }
    },
    []
  );

  // Start interview session
  const startInterviewSession = useCallback(
    (type?: InterviewType) => {
      if (type) setInterviewType(type);
      const startingIndex = type === 'coding' ? 3 : 0;
      setCurrentQuestionIndex(startingIndex);
      setElapsedSeconds(0);
      setIsFollowUpActive(false);

      const firstQ = questions[startingIndex];
      const introText =
        startingIndex === 3
          ? firstQ.questionText
          : `Hello Alex, welcome to your interview with Nova Technologies. ${firstQ.questionText}`;

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
    [questions, speakQuestion]
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

  // End Interview Session
  const endInterviewSession = useCallback(() => {
    const closingText =
      'Great, that wraps up everything for today’s session. Your interview is complete, and your comprehensive evaluation report is ready for review.';
    setInterviewState('completed');
    setAvatarState('speaking');
    setActiveSubtitle(closingText);

    // Save final report with actual elapsed duration
    setLatestReport((prev) => ({
      ...prev,
      durationSeconds: Math.max(elapsedSeconds, 900),
    }));

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(closingText);
      window.speechSynthesis.speak(utterance);
    }
  }, [elapsedSeconds]);

  // Reset Session
  const resetSession = useCallback(() => {
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
        dismissSuggestion,
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
