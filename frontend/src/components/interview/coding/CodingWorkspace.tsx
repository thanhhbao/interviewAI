'use client';

import React, { useState } from 'react';
import dynamic from 'next/dynamic';
import {
  Play,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Sparkles,
  HelpCircle,
  Terminal,
  Code2,
  Check,
  ChevronRight,
  Maximize2,
  FileCode,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { PALETTE } from '@/lib/motion';
import { InterviewAvatar3D } from '../InterviewAvatar3D';

// Dynamically import Monaco Editor to avoid SSR issues
const Editor = dynamic(() => import('@monaco-editor/react'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-neutral-950 text-neutral-400 text-xs font-mono">
      Initializing Monaco Code Editor...
    </div>
  ),
});

export const CodingWorkspace: React.FC = () => {
  const {
    currentQuestion,
    avatarState,
    activeCode,
    setActiveCode,
    activeLanguage,
    setActiveLanguage,
    testResults,
    isRunningCode,
    runCodeAction,
    submitCodingSolution,
    askForClarification,
    interviewState,
  } = useInterview();

  const [activeTab, setActiveTab] = useState<'tests' | 'console'>('tests');
  const [selectedTestCaseIndex, setSelectedTestCaseIndex] = useState<number>(0);

  const codingTask = currentQuestion?.codingTask;
  if (!codingTask) return null;

  const handleResetCode = () => {
    if (codingTask.starterCode[activeLanguage]) {
      setActiveCode(codingTask.starterCode[activeLanguage]);
    }
  };

  const handleLanguageChange = (lang: string) => {
    setActiveLanguage(lang);
    if (codingTask.starterCode[lang]) {
      setActiveCode(codingTask.starterCode[lang]);
    }
  };

  return (
    <div className="w-full h-[calc(100vh-64px)] flex overflow-hidden" style={{ backgroundColor: PALETTE.bg }}>
      {/* LEFT PANE: AI Interviewer & Problem Description */}
      <div
        className="w-2/5 flex flex-col h-full overflow-hidden"
        style={{ borderRight: `1px solid ${PALETTE.border}`, backgroundColor: PALETTE.surface }}
      >
        {/* Compact AI Interviewer Bar */}
        <div
          className="p-4 flex items-center gap-4"
          style={{ borderBottom: `1px solid ${PALETTE.border}`, backgroundColor: 'rgba(255,255,255,0.015)' }}
        >
          <div
            className="w-24 h-24 rounded-2xl overflow-hidden shrink-0"
            style={{ backgroundColor: 'rgba(255,255,255,0.03)', border: `1px solid ${PALETTE.border}` }}
          >
            <InterviewAvatar3D state={avatarState} isCompact={true} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: PALETTE.accentBright }} />
              <span className="text-xs font-semibold" style={{ color: PALETTE.textPrimary }}>
                AI Interviewer
              </span>
              <span className="text-[10px] capitalize" style={{ color: PALETTE.textSecondary }}>
                · {avatarState}
              </span>
            </div>
            <p className="text-xs mt-1 line-clamp-2" style={{ color: '#c7cac5' }}>
              &ldquo;Take your time to structure your solution. Feel free to ask if you need
              clarifications.&rdquo;
            </p>
            <button
              onClick={askForClarification}
              className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium transition-colors hover:opacity-80"
              style={{ backgroundColor: 'rgba(85,229,207,0.1)', color: PALETTE.accentBright }}
            >
              <HelpCircle className="w-3 h-3" />
              <span>Ask for Clarification</span>
            </button>
          </div>
        </div>

        {/* Problem Specification Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span
                className="px-2 py-0.5 rounded text-[11px] font-medium"
                style={{ backgroundColor: 'rgba(85,229,207,0.1)', color: PALETTE.accentBright }}
              >
                Easy / Medium
              </span>
              <span className="text-xs" style={{ color: PALETTE.textSecondary }}>Target Time: 15 mins</span>
            </div>
            <h2 className="text-lg font-semibold" style={{ color: PALETTE.textPrimary }}>
              {codingTask.problemTitle}
            </h2>
            <p className="text-xs mt-2 leading-relaxed whitespace-pre-line" style={{ color: '#c7cac5' }}>
              {codingTask.problemDescription}
            </p>
          </div>

          {/* Requirements & Constraints */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
              Requirements
            </h4>
            <ul className="space-y-1.5 text-xs" style={{ color: '#c7cac5' }}>
              {codingTask.requirements.map((req, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0" style={{ backgroundColor: PALETTE.accent }} />
                  <span>{req}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Examples */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
              Examples
            </h4>
            <div className="space-y-2.5">
              {codingTask.examples.map((ex, i) => (
                <div
                  key={i}
                  className="p-3 rounded-xl font-mono text-xs space-y-1"
                  style={{ backgroundColor: 'rgba(255,255,255,0.025)', border: `1px solid ${PALETTE.border}` }}
                >
                  <p style={{ color: '#c7cac5' }}>
                    <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>
                      Input:
                    </span>{' '}
                    {ex.input}
                  </p>
                  <p style={{ color: '#c7cac5' }}>
                    <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>
                      Output:
                    </span>{' '}
                    {ex.output}
                  </p>
                  {ex.explanation && (
                    <p className="text-[11px] font-sans mt-1" style={{ color: PALETTE.textSecondary }}>
                      {ex.explanation}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Constraints */}
          <div className="space-y-3 pb-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
              Constraints
            </h4>
            <ul className="space-y-1 text-xs font-mono" style={{ color: PALETTE.textSecondary }}>
              {codingTask.constraints.map((c, i) => (
                <li key={i} className="flex items-center gap-2">
                  <span style={{ color: '#5a615c' }}>•</span>
                  <span>{c}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* RIGHT PANE: Code Editor & Execution Results */}
      <div className="flex-1 flex flex-col h-full overflow-hidden" style={{ backgroundColor: '#0A1110' }}>
        {/* Editor Toolbar */}
        <div
          className="px-4 py-2.5 flex items-center justify-between shrink-0"
          style={{ backgroundColor: PALETTE.surface2, borderBottom: `1px solid ${PALETTE.border}` }}
        >
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs font-medium" style={{ color: '#c7cac5' }}>
              <Code2 className="w-4 h-4" style={{ color: PALETTE.accent }} />
              <span>Solution</span>
            </div>

            {/* Language Selector */}
            <select
              value={activeLanguage}
              onChange={(e) => handleLanguageChange(e.target.value)}
              className="text-xs rounded-lg px-2.5 py-1 outline-none transition-colors"
              style={{ backgroundColor: 'rgba(255,255,255,0.04)', color: '#c7cac5', border: `1px solid ${PALETTE.border}` }}
            >
              <option value="python">Python 3</option>
              <option value="typescript">TypeScript</option>
              <option value="javascript">JavaScript</option>
            </select>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleResetCode}
              className="p-1.5 rounded-lg transition-colors hover:bg-white/[0.06]"
              style={{ color: PALETTE.textSecondary }}
              title="Reset code"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>

            <button
              onClick={runCodeAction}
              disabled={isRunningCode}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-50 hover:bg-white/[0.06]"
              style={{ backgroundColor: 'rgba(255,255,255,0.04)', color: '#c7cac5', border: `1px solid ${PALETTE.border}` }}
            >
              <Play className="w-3 h-3 fill-current" style={{ color: PALETTE.accentBright }} />
              <span>{isRunningCode ? 'Running...' : 'Run Code'}</span>
            </button>

            <button
              onClick={submitCodingSolution}
              disabled={interviewState === 'reviewing_code'}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-medium shadow-sm transition-opacity hover:opacity-90 disabled:opacity-50"
              style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
            >
              <Check className="w-3.5 h-3.5" />
              <span>Submit Solution</span>
            </button>
          </div>
        </div>

        {/* Monaco Editor Pane */}
        <div className="flex-1 min-h-[300px]">
          <Editor
            height="100%"
            language={activeLanguage === 'typescript' ? 'typescript' : activeLanguage}
            theme="vs-dark"
            value={activeCode}
            onChange={(val) => setActiveCode(val || '')}
            options={{
              minimap: { enabled: false },
              fontSize: 13,
              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              lineNumbers: 'on',
              scrollBeyondLastLine: false,
              automaticLayout: true,
              tabSize: 4,
              padding: { top: 12, bottom: 12 },
              renderLineHighlight: 'all',
            }}
          />
        </div>

        {/* Bottom Console / Test Results Panel */}
        <div className="h-60 flex flex-col shrink-0" style={{ borderTop: `1px solid ${PALETTE.border}`, backgroundColor: '#080D0C' }}>
          {/* Panel Tabs */}
          <div
            className="px-4 py-2 flex items-center justify-between"
            style={{ backgroundColor: 'rgba(255,255,255,0.02)', borderBottom: `1px solid ${PALETTE.border}` }}
          >
            <div className="flex items-center gap-4 text-xs">
              <button
                onClick={() => setActiveTab('tests')}
                className="flex items-center gap-1.5 font-medium transition-colors"
                style={{ color: activeTab === 'tests' ? PALETTE.textPrimary : PALETTE.textSecondary }}
              >
                <CheckCircle2 className="w-3.5 h-3.5" style={{ color: PALETTE.accent }} />
                <span>Test Cases</span>
                {testResults && (
                  <span
                    className="text-[10px] px-1.5 py-0.2 rounded-full font-mono"
                    style={
                      testResults.failed === 0
                        ? { backgroundColor: 'rgba(85,229,207,0.15)', color: PALETTE.accentBright }
                        : { backgroundColor: 'rgba(248,113,113,0.15)', color: '#f87171' }
                    }
                  >
                    {testResults.passed}/{testResults.total} passed
                  </span>
                )}
              </button>

              <button
                onClick={() => setActiveTab('console')}
                className="flex items-center gap-1.5 font-medium transition-colors"
                style={{ color: activeTab === 'console' ? PALETTE.textPrimary : PALETTE.textSecondary }}
              >
                <Terminal className="w-3.5 h-3.5" style={{ color: PALETTE.textSecondary }} />
                <span>Console Output</span>
              </button>
            </div>

            {testResults && (
              <span className="text-[11px] font-mono" style={{ color: PALETTE.textSecondary }}>
                Runtime: {testResults.totalRuntimeMs}ms
              </span>
            )}
          </div>

          {/* Tab Content */}
          <div className="flex-1 p-4 overflow-y-auto font-mono text-xs" style={{ color: '#c7cac5' }}>
            {activeTab === 'tests' ? (
              <div className="h-full flex flex-col">
                {/* Test case buttons */}
                <div className="flex items-center gap-2 mb-3">
                  {codingTask.testCases.map((tc, index) => {
                    const res = testResults?.results.find((r) => r.testId === tc.id);
                    const isPassed = res ? res.passed : null;
                    const isSelected = selectedTestCaseIndex === index;

                    return (
                      <button
                        key={tc.id}
                        onClick={() => setSelectedTestCaseIndex(index)}
                        className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs transition-colors"
                        style={
                          isSelected
                            ? { backgroundColor: 'rgba(255,255,255,0.06)', color: PALETTE.textPrimary, fontWeight: 500, border: `1px solid ${PALETTE.border}` }
                            : { backgroundColor: 'rgba(255,255,255,0.02)', color: PALETTE.textSecondary }
                        }
                      >
                        {isPassed === true && (
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: PALETTE.accentBright }} />
                        )}
                        {isPassed === false && (
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: '#f87171' }} />
                        )}
                        <span>Case {index + 1}</span>
                      </button>
                    );
                  })}
                </div>

                {/* Selected Test Case Details */}
                {(() => {
                  const tc = codingTask.testCases[selectedTestCaseIndex];
                  const res = testResults?.results.find((r) => r.testId === tc?.id);

                  return (
                    <div
                      className="space-y-2 p-3 rounded-xl"
                      style={{ backgroundColor: 'rgba(255,255,255,0.02)', border: `1px solid ${PALETTE.border}` }}
                    >
                      <div>
                        <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>Input:</span>
                        <p className="mt-0.5" style={{ color: '#dcded9' }}>{tc?.input}</p>
                      </div>

                      <div>
                        <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>Expected:</span>
                        <p className="mt-0.5" style={{ color: '#dcded9' }}>{tc?.expected}</p>
                      </div>

                      {res && (
                        <div>
                          <span className="text-[11px]" style={{ color: res.passed ? PALETTE.accentBright : '#f87171' }}>
                            Received:
                          </span>
                          <p
                            className="mt-0.5"
                            style={{ color: res.passed ? PALETTE.accent : '#fca5a5', fontWeight: res.passed ? 400 : 600 }}
                          >
                            {res.received}
                          </p>
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            ) : (
              <div className="space-y-1 font-mono text-xs" style={{ color: PALETTE.textSecondary }}>
                <p style={{ color: PALETTE.accentBright }}>=== Sandboxed execution environment initialized ===</p>
                {testResults ? (
                  testResults.results.map((r, i) => (
                    <p key={i} style={{ color: r.passed ? '#c7cac5' : '#fca5a5' }}>
                      [Test {r.testId}]: {r.passed ? 'PASSED' : `FAILED (${r.error || 'Assertion'})`} in{' '}
                      {r.runtimeMs}ms
                    </p>
                  ))
                ) : (
                  <p>Click &ldquo;Run Code&rdquo; to execute tests and view console output.</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
