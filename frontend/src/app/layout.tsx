import './globals.css';
import type { Metadata } from 'next';
import { InterviewProvider } from '@/context/InterviewContext';
import { SiteChrome } from '@/components/layout/SiteChrome';
import { GeistSans } from "geist/font/sans";
import { cn } from "@/lib/utils";

const geist = GeistSans;

export const metadata: Metadata = {
  title: 'InterviewAI — Modern AI Interview Assistant',
  description:
    'Prepare for technical and behavioral interviews with an interactive 3D AI interviewer, coding workspace, and structured reports.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={cn("font-sans", geist.variable)}>
      <body className="min-h-screen flex bg-[#FAF9F7] dark:bg-[#09090B] text-neutral-900 dark:text-neutral-100">
        <InterviewProvider>
          <div className="flex w-full min-h-screen">
            <SiteChrome>{children}</SiteChrome>
          </div>
        </InterviewProvider>
      </body>
    </html>
  );
}
