'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  FileText,
  Upload,
  Sparkles,
  Edit3,
  Briefcase,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Plus,
  RefreshCw,
  Award,
  BookOpen,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { TopHeader } from '@/components/layout/TopHeader';
import { ResumeOptimizerModal } from '@/components/resume/ResumeOptimizerModal';

export default function ResumeJobPage() {
  const router = useRouter();
  const { candidate, targetJob, startInterviewSession } = useInterview();
  const [isOptimizerOpen, setIsOptimizerOpen] = useState(false);
  const [isEditingJob, setIsEditingJob] = useState(false);

  const handleStartPractice = () => {
    router.push('/setup');
  };

  return (
    <div className="flex-1 flex flex-col min-h-screen">
      <TopHeader
        title="Resume & Target Job"
        subtitle="Review and align your background with your target position"
      />

      <main className="flex-1 max-w-6xl w-full mx-auto px-8 py-8 space-y-10">
        {/* TWO COLUMN SECTION */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* LEFT: YOUR RESUME */}
          <div className="p-7 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 shadow-xs space-y-6 flex flex-col justify-between">
            <div className="space-y-6">
              {/* Header */}
              <div className="flex items-center justify-between pb-3 border-b border-neutral-100 dark:border-neutral-800/60">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-neutral-100 dark:bg-neutral-800 text-neutral-800 dark:text-neutral-200 flex items-center justify-center">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div>
                    <h2 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                      Your Resume
                    </h2>
                    <span className="text-[11px] text-neutral-400 font-mono">
                      {candidate.resumeFileName}
                    </span>
                  </div>
                </div>

                <button
                  onClick={() => setIsOptimizerOpen(true)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-50 dark:bg-indigo-950/70 text-indigo-600 dark:text-indigo-400 text-xs font-semibold hover:bg-indigo-100 dark:hover:bg-indigo-900/80 transition-colors shadow-xs"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Optimize Resume</span>
                </button>
              </div>

              {/* Candidate Info */}
              <div className="space-y-1.5">
                <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">
                  {candidate.name}
                </h3>
                <p className="text-xs text-indigo-600 dark:text-indigo-400 font-medium">
                  {candidate.title}
                </p>
                <p className="text-xs text-neutral-600 dark:text-neutral-300 pt-1 leading-relaxed">
                  {candidate.summary}
                </p>
              </div>

              {/* Work Experience */}
              <div className="space-y-3">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
                  Work Experience
                </span>
                <div className="space-y-3">
                  {candidate.experience.map((exp) => (
                    <div
                      key={exp.id}
                      className="p-3.5 rounded-2xl bg-neutral-50/70 dark:bg-neutral-900/50 border border-neutral-200/60 dark:border-neutral-800/60 space-y-1.5 text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-neutral-900 dark:text-neutral-100">
                          {exp.role}
                        </span>
                        <span className="text-[11px] text-neutral-400">{exp.duration}</span>
                      </div>
                      <span className="text-neutral-500 block text-[11px] font-medium">
                        {exp.company}
                      </span>
                      <ul className="space-y-1 text-neutral-600 dark:text-neutral-400 pt-1 text-[11px]">
                        {exp.highlights.slice(0, 2).map((h, i) => (
                          <li key={i} className="flex items-start gap-1.5">
                            <span className="text-neutral-300 dark:text-neutral-600">•</span>
                            <span className="line-clamp-2">{h}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>

              {/* Skills */}
              <div className="space-y-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
                  Key Skills
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {candidate.skills.map((skill) => (
                    <span
                      key={skill}
                      className="px-2.5 py-1 rounded-lg bg-neutral-100 dark:bg-neutral-800 text-[11px] font-medium text-neutral-700 dark:text-neutral-300"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>

              {/* Education */}
              <div className="space-y-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
                  Education
                </span>
                {candidate.education.map((edu) => (
                  <div key={edu.id} className="text-xs text-neutral-600 dark:text-neutral-400">
                    <span className="font-medium text-neutral-900 dark:text-neutral-200">
                      {edu.degree}
                    </span>{' '}
                    — {edu.school} ({edu.year})
                  </div>
                ))}
              </div>
            </div>

            {/* Bottom Actions for Resume */}
            <div className="flex items-center gap-3 pt-4 border-t border-neutral-100 dark:border-neutral-800/60 text-xs">
              <button
                onClick={() => setIsOptimizerOpen(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800 text-neutral-700 dark:text-neutral-300 transition-colors"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Replace Resume</span>
              </button>
              <button
                onClick={() => setIsOptimizerOpen(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800 text-neutral-700 dark:text-neutral-300 transition-colors"
              >
                <Edit3 className="w-3.5 h-3.5" />
                <span>Edit Profile</span>
              </button>
            </div>
          </div>

          {/* RIGHT: TARGET JOB */}
          <div className="p-7 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 shadow-xs space-y-6 flex flex-col justify-between">
            <div className="space-y-6">
              {/* Header */}
              <div className="flex items-center justify-between pb-3 border-b border-neutral-100 dark:border-neutral-800/60">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-neutral-100 dark:bg-neutral-800 text-neutral-800 dark:text-neutral-200 flex items-center justify-center">
                    <Briefcase className="w-4 h-4" />
                  </div>
                  <div>
                    <h2 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                      Target Job Description
                    </h2>
                    <span className="text-[11px] text-neutral-400">
                      {targetJob.company} · {targetJob.location}
                    </span>
                  </div>
                </div>

                <span className="px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/70 text-emerald-600 dark:text-emerald-400 text-xs font-semibold">
                  Active
                </span>
              </div>

              {/* Job Title & Overview */}
              <div className="space-y-1.5">
                <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">
                  {targetJob.title}
                </h3>
                <p className="text-xs text-neutral-600 dark:text-neutral-300 leading-relaxed">
                  {targetJob.description}
                </p>
              </div>

              {/* Required Skills */}
              <div className="space-y-2.5">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
                  Required Competencies
                </span>
                <ul className="space-y-1.5 text-xs text-neutral-700 dark:text-neutral-300">
                  {targetJob.requiredSkills.map((req, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 mt-0.5 shrink-0" />
                      <span className="leading-tight">{req}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Preferred Skills */}
              <div className="space-y-2.5">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
                  Preferred / Nice-to-Have
                </span>
                <ul className="space-y-1.5 text-xs text-neutral-600 dark:text-neutral-400">
                  {targetJob.preferredSkills.map((pref, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1.5 shrink-0"></span>
                      <span className="leading-tight">{pref}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Bottom Actions for Job */}
            <div className="flex items-center gap-3 pt-4 border-t border-neutral-100 dark:border-neutral-800/60 text-xs">
              <button
                onClick={() => setIsEditingJob(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800 text-neutral-700 dark:text-neutral-300 transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add Job Description</span>
              </button>
              <button
                onClick={() => setIsEditingJob(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800 text-neutral-700 dark:text-neutral-300 transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Replace Job</span>
              </button>
            </div>
          </div>
        </div>

        {/* JOB MATCH ANALYSIS SECTION */}
        <section className="p-8 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 shadow-xs space-y-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-neutral-100 dark:border-neutral-800/60">
            <div className="space-y-1">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
                AI Context Synthesis
              </span>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Job Match Analysis
              </h3>
            </div>

            {/* Match Score Badge */}
            <div className="flex items-center gap-3">
              <div className="text-right">
                <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                  {targetJob.matchScore}%
                </span>
                <span className="block text-[11px] text-emerald-600 dark:text-emerald-400 font-medium">
                  Strong Match Potential
                </span>
              </div>
              <div className="w-12 h-12 rounded-2xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center">
                <Award className="w-6 h-6" />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
            {/* Matched Skills */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                <h4 className="text-xs font-semibold text-neutral-800 dark:text-neutral-200">
                  Matched Skills ({targetJob.matchedSkills.length})
                </h4>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {targetJob.matchedSkills.map((s) => (
                  <span
                    key={s}
                    className="px-2.5 py-1 rounded-lg bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 text-xs font-medium"
                  >
                    ✓ {s}
                  </span>
                ))}
              </div>
            </div>

            {/* Partially Matched / Missing */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                <h4 className="text-xs font-semibold text-neutral-800 dark:text-neutral-200">
                  Growth & Missing Skills
                </h4>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {targetJob.missingSkills.map((s) => (
                  <span
                    key={s}
                    className="px-2.5 py-1 rounded-lg bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 text-xs font-medium"
                  >
                    ! {s}
                  </span>
                ))}
              </div>
            </div>

            {/* Suggested Focus Areas */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
                <h4 className="text-xs font-semibold text-neutral-800 dark:text-neutral-200">
                  Suggested Interview Focus
                </h4>
              </div>
              <ul className="space-y-1.5 text-xs text-neutral-600 dark:text-neutral-300">
                {targetJob.focusAreas.map((area, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-indigo-500 font-bold">•</span>
                    <span>{area}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Primary CTA */}
          <div className="pt-6 border-t border-neutral-100 dark:border-neutral-800/60 flex items-center justify-between">
            <p className="text-xs text-neutral-500">
              Your resume and target job context are loaded into your AI interviewer.
            </p>
            <button
              onClick={handleStartPractice}
              className="px-6 py-3 rounded-2xl bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-100 text-white dark:text-neutral-950 text-xs font-semibold shadow-md flex items-center gap-2 transition-all"
            >
              <span>Practice Interview for This Role</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </section>
      </main>

      {/* Resume Optimizer Drawer/Modal */}
      <ResumeOptimizerModal isOpen={isOptimizerOpen} onClose={() => setIsOptimizerOpen(false)} />
    </div>
  );
}
