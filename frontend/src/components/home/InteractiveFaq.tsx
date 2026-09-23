'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus } from 'lucide-react';
import { PALETTE, fadeUp, viewportOnce } from '@/lib/motion';

interface FaqItem {
  question: string;
  answer: string;
}

const faqs: FaqItem[] = [
  {
    question: 'How realistic is the AI voice interviewer compared to a human recruiter?',
    answer:
      'Our conversational engine operates with sub-300ms latency, enabling natural interruptions and cadence. It is trained on rubrics from top tech firms (FAANG/MAMAA), asking pointed follow-up questions tailored to your explanations rather than following static scripts.',
  },
  {
    question: 'Can I execute code during interviews?',
    answer:
      'Yes. Our built-in Monaco editor supports TypeScript, Python, Java, C++, and Go. Code is evaluated against automated test suites testing both standard cases and tricky edge cases, paired with Big-O time and space complexity estimations.',
  },
  {
    question: 'How does resume matching work?',
    answer:
      'Upload any PDF resume and paste the target job description or link. The engine extracts exact skill overlaps and deficits, dynamically biasing the interview questions toward your target seniority and potential gaps.',
  },
  {
    question: 'What kind of report do I receive after finishing an interview session?',
    answer:
      'You receive an exhaustive diagnostic report with composite hiring scores (0-100), STAR structure compliance, speech metrics (WPM, filler word count), line-by-line transcript commentary, and optimized answer examples.',
  },
];

export const InteractiveFaq: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  // FAQ is an intentional visual rest point — the act's own StarsDataField
  // is already deliberately sparser back here; no extra decoration needed.
  return (
    <section className="relative w-full py-24 md:py-32 overflow-hidden">
      <div className="max-w-[720px] mx-auto px-6 relative">
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={viewportOnce} className="mb-12 space-y-3">
          <p className="text-[11px] font-mono tracking-[0.12em]" style={{ color: PALETTE.accent }}>
            FAQ
          </p>
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
            Everything you need to know.
          </h2>
        </motion.div>

        <div>
          {faqs.map((faq, index) => {
            const isOpen = openIndex === index;
            return (
              <motion.div
                key={index}
                variants={fadeUp}
                initial="hidden"
                whileInView="show"
                viewport={viewportOnce}
                style={{ borderTop: `1px solid ${PALETTE.borderTeal}` }}
              >
                <button
                  onClick={() => setOpenIndex(isOpen ? null : index)}
                  className="w-full py-6 text-left flex items-center justify-between gap-6 select-none group"
                  aria-expanded={isOpen}
                >
                  <span
                    className="text-base md:text-lg font-medium transition-colors"
                    style={{ color: isOpen ? PALETTE.textPrimary : '#c7cac5' }}
                  >
                    {faq.question}
                  </span>
                  <motion.span
                    animate={{ rotate: isOpen ? 45 : 0 }}
                    transition={{ duration: 0.25, ease: 'easeOut' }}
                    className="shrink-0"
                    style={{ color: isOpen ? PALETTE.accentBright : PALETTE.textSecondary }}
                  >
                    <Plus className="w-4 h-4" />
                  </motion.span>
                </button>

                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                      className="overflow-hidden"
                    >
                      <p className="pb-6 text-sm leading-relaxed max-w-xl" style={{ color: PALETTE.textSecondary }}>
                        {faq.answer}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
          <div style={{ borderTop: `1px solid ${PALETTE.borderTeal}` }} />
        </div>
      </div>
    </section>
  );
};
