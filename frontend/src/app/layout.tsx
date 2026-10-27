import './globals.css';
import type { Metadata } from 'next';
import { InterviewProvider } from '@/context/InterviewContext';
import { AuthProvider } from '@/context/AuthContext';
import { LoginModal } from '@/components/auth/LoginModal';
import { SiteChrome } from '@/components/layout/SiteChrome';
import { Preloader } from '@/components/layout/Preloader';
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
      <body
        className="min-h-screen flex bg-[#FAF9F7] dark:bg-[#09090B] text-neutral-900 dark:text-neutral-100"
        data-gramm="false"
        data-gramm_editor="false"
        data-enable-grammarly="false"
      >
        <AuthProvider>
          <InterviewProvider>
            <Preloader />
            <LoginModal />
            <div className="flex w-full min-h-screen">
              <SiteChrome>{children}</SiteChrome>
            </div>
          </InterviewProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
