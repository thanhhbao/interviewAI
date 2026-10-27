'use client';

import React, { useEffect, useState } from 'react';
import { Plus, Trash2, Check } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { CandidateProfile, ResumeWorkExperience, ResumeEducation, ResumeProject } from '@/lib/types/interview';
import { ModalShell } from '@/components/app/ModalShell';
import { PALETTE } from '@/lib/motion';

interface EditProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
}

let expCounter = 0;
let eduCounter = 0;
let projCounter = 0;

// Separate from ResumeOptimizerModal on purpose — that one is for AI
// suggestions on existing content, this one is direct manual editing of the
// candidate profile. Saving here doesn't require re-uploading a resume.
export const EditProfileModal: React.FC<EditProfileModalProps> = ({ isOpen, onClose }) => {
  const { candidate, updateCandidateProfile } = useInterview();
  const [draft, setDraft] = useState<CandidateProfile>(candidate);
  const [skillsText, setSkillsText] = useState(candidate.skills.join(', '));
  const [savedNotice, setSavedNotice] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setDraft(candidate);
      setSkillsText(candidate.skills.join(', '));
      setSavedNotice(false);
    }
  }, [isOpen, candidate]);

  if (!isOpen) return null;

  const updateExperience = (id: string, patch: Partial<ResumeWorkExperience>) => {
    setDraft((prev) => ({
      ...prev,
      experience: prev.experience.map((exp) => (exp.id === id ? { ...exp, ...patch } : exp)),
    }));
  };

  const updateHighlight = (expId: string, index: number, value: string) => {
    setDraft((prev) => ({
      ...prev,
      experience: prev.experience.map((exp) =>
        exp.id === expId
          ? { ...exp, highlights: exp.highlights.map((h, i) => (i === index ? value : h)) }
          : exp
      ),
    }));
  };

  const addExperience = () => {
    expCounter += 1;
    setDraft((prev) => ({
      ...prev,
      experience: [
        ...prev.experience,
        { id: `exp-new-${expCounter}`, company: '', role: '', duration: '', highlights: [''] },
      ],
    }));
  };

  const removeExperience = (id: string) => {
    setDraft((prev) => ({ ...prev, experience: prev.experience.filter((exp) => exp.id !== id) }));
  };

  const updateEducation = (id: string, patch: Partial<ResumeEducation>) => {
    setDraft((prev) => ({
      ...prev,
      education: prev.education.map((edu) => (edu.id === id ? { ...edu, ...patch } : edu)),
    }));
  };

  const addEducation = () => {
    eduCounter += 1;
    setDraft((prev) => ({
      ...prev,
      education: [...prev.education, { id: `edu-new-${eduCounter}`, school: '', degree: '', year: '' }],
    }));
  };

  const removeEducation = (id: string) => {
    setDraft((prev) => ({ ...prev, education: prev.education.filter((edu) => edu.id !== id) }));
  };

  const updateProject = (id: string, patch: Partial<ResumeProject>) => {
    setDraft((prev) => ({
      ...prev,
      projects: prev.projects.map((p) => (p.id === id ? { ...p, ...patch } : p)),
    }));
  };

  const addProject = () => {
    projCounter += 1;
    setDraft((prev) => ({
      ...prev,
      projects: [
        ...prev.projects,
        { id: `proj-new-${projCounter}`, name: '', description: '', tech: [] },
      ],
    }));
  };

  const removeProject = (id: string) => {
    setDraft((prev) => ({ ...prev, projects: prev.projects.filter((p) => p.id !== id) }));
  };

  const handleSave = () => {
    const updated: CandidateProfile = {
      ...draft,
      skills: skillsText
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean),
    };
    updateCandidateProfile(updated);
    setSavedNotice(true);
    setTimeout(() => {
      setSavedNotice(false);
      onClose();
    }, 900);
  };

  const inputClass =
    'w-full rounded-[8px] px-3 py-2 outline-none transition-colors text-xs';
  const inputStyle = {
    backgroundColor: PALETTE.surface2,
    border: `1px solid ${PALETTE.border}`,
    color: PALETTE.textPrimary,
  };

  return (
    <ModalShell
      title="Edit Profile"
      onClose={onClose}
      maxWidthClassName="max-w-3xl"
      headerExtra={
        savedNotice ? (
          <span className="flex items-center gap-1 text-[11px] font-medium" style={{ color: PALETTE.accent }}>
            <Check className="w-3.5 h-3.5" />
            Profile updated
          </span>
        ) : undefined
      }
      footer={
        <>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-[8px] text-xs font-medium transition-colors hover:bg-white/[0.05]"
            style={{ color: PALETTE.textSecondary }}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 rounded-[8px] text-xs font-semibold"
            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
          >
            Save Profile
          </button>
        </>
      }
    >
        {/* Form body */}
        <div className="p-6 space-y-6 text-xs">
          {/* Basics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="block" style={{ color: PALETTE.textSecondary }}>Full Name</label>
              <input
                className={inputClass}
                style={inputStyle}
                value={draft.name}
                onChange={(e) => setDraft((p) => ({ ...p, name: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <label className="block" style={{ color: PALETTE.textSecondary }}>Current / Target Title</label>
              <input
                className={inputClass}
                style={inputStyle}
                value={draft.title}
                onChange={(e) => setDraft((p) => ({ ...p, title: e.target.value }))}
              />
            </div>
          </div>

          <div className="space-y-1">
            <label className="block" style={{ color: PALETTE.textSecondary }}>Professional Summary</label>
            <textarea
              rows={3}
              className={`${inputClass} leading-relaxed`}
              style={inputStyle}
              value={draft.summary}
              onChange={(e) => setDraft((p) => ({ ...p, summary: e.target.value }))}
            />
          </div>

          <div className="space-y-1">
            <label className="block" style={{ color: PALETTE.textSecondary }}>Skills (comma-separated)</label>
            <input className={inputClass} value={skillsText} onChange={(e) => setSkillsText(e.target.value)} />
          </div>

          {/* Experience */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
                Work Experience
              </span>
              <button
                onClick={addExperience}
                className="flex items-center gap-1 font-medium hover:underline" style={{ color: PALETTE.accent }}
              >
                <Plus className="w-3.5 h-3.5" /> Add
              </button>
            </div>
            {draft.experience.map((exp) => (
              <div
                key={exp.id}
                className="p-3.5 rounded-[10px] space-y-2"
                style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
              >
                <div className="flex items-center gap-2">
                  <input
                    className={inputClass}
                style={inputStyle}
                    placeholder="Role"
                    value={exp.role}
                    onChange={(e) => updateExperience(exp.id, { role: e.target.value })}
                  />
                  <input
                    className={inputClass}
                style={inputStyle}
                    placeholder="Company"
                    value={exp.company}
                    onChange={(e) => updateExperience(exp.id, { company: e.target.value })}
                  />
                  <input
                    className={inputClass}
                style={inputStyle}
                    placeholder="Duration"
                    value={exp.duration}
                    onChange={(e) => updateExperience(exp.id, { duration: e.target.value })}
                  />
                  <button
                    onClick={() => removeExperience(exp.id)}
                    className="p-2 rounded-[8px] shrink-0 transition-colors hover:bg-white/[0.06]" style={{ color: PALETTE.textSecondary }}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
                {exp.highlights.map((h, i) => (
                  <textarea
                    key={i}
                    rows={2}
                    className={`${inputClass} leading-relaxed`}
              style={inputStyle}
                    value={h}
                    onChange={(e) => updateHighlight(exp.id, i, e.target.value)}
                  />
                ))}
              </div>
            ))}
          </div>

          {/* Education */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
                Education
              </span>
              <button
                onClick={addEducation}
                className="flex items-center gap-1 font-medium hover:underline" style={{ color: PALETTE.accent }}
              >
                <Plus className="w-3.5 h-3.5" /> Add
              </button>
            </div>
            {draft.education.map((edu) => (
              <div key={edu.id} className="flex items-center gap-2">
                <input
                  className={inputClass}
                style={inputStyle}
                  placeholder="Degree"
                  value={edu.degree}
                  onChange={(e) => updateEducation(edu.id, { degree: e.target.value })}
                />
                <input
                  className={inputClass}
                style={inputStyle}
                  placeholder="School"
                  value={edu.school}
                  onChange={(e) => updateEducation(edu.id, { school: e.target.value })}
                />
                <input
                  className={inputClass}
                style={inputStyle}
                  placeholder="Year"
                  value={edu.year}
                  onChange={(e) => updateEducation(edu.id, { year: e.target.value })}
                />
                <button
                  onClick={() => removeEducation(edu.id)}
                  className="p-2 rounded-[8px] shrink-0 transition-colors hover:bg-white/[0.06]" style={{ color: PALETTE.textSecondary }}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>

          {/* Projects */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
                Projects
              </span>
              <button
                onClick={addProject}
                className="flex items-center gap-1 font-medium hover:underline" style={{ color: PALETTE.accent }}
              >
                <Plus className="w-3.5 h-3.5" /> Add
              </button>
            </div>
            {draft.projects.map((proj) => (
              <div
                key={proj.id}
                className="p-3.5 rounded-[10px] space-y-2"
                style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
              >
                <div className="flex items-center gap-2">
                  <input
                    className={inputClass}
                style={inputStyle}
                    placeholder="Project name"
                    value={proj.name}
                    onChange={(e) => updateProject(proj.id, { name: e.target.value })}
                  />
                  <button
                    onClick={() => removeProject(proj.id)}
                    className="p-2 rounded-[8px] shrink-0 transition-colors hover:bg-white/[0.06]" style={{ color: PALETTE.textSecondary }}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
                <textarea
                  rows={2}
                  className={`${inputClass} leading-relaxed`}
              style={inputStyle}
                  placeholder="Description"
                  value={proj.description}
                  onChange={(e) => updateProject(proj.id, { description: e.target.value })}
                />
                <input
                  className={inputClass}
                style={inputStyle}
                  placeholder="Tech (comma-separated)"
                  value={proj.tech.join(', ')}
                  onChange={(e) =>
                    updateProject(proj.id, {
                      tech: e.target.value.split(',').map((t) => t.trim()).filter(Boolean),
                    })
                  }
                />
              </div>
            ))}
          </div>
        </div>
    </ModalShell>
  );
};
