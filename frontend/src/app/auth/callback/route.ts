import { NextResponse } from 'next/server';
import type { EmailOtpType } from '@supabase/supabase-js';
import { createClient } from '@/lib/supabase/server';

// Handles:
// - Email confirmation / magic link (token_hash + type)
// - GitHub OAuth redirect (code)
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const next = searchParams.get('next') ?? '/';

  // Email OTP / confirmation link sends token_hash + type (not code)
  const token_hash = searchParams.get('token_hash');
  const type = searchParams.get('type') as EmailOtpType | null;
  if (token_hash && type) {
    const supabase = await createClient();
    const { error } = await supabase.auth.verifyOtp({ token_hash, type });
    if (!error) {
      return NextResponse.redirect(`${origin}${next}`);
    }
  }

  // OAuth (GitHub etc.) sends a code via PKCE
  const code = searchParams.get('code');
  if (code) {
    const supabase = await createClient();
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (!error) {
      return NextResponse.redirect(`${origin}${next}`);
    }
  }

  return NextResponse.redirect(`${origin}/login?error=auth_callback_failed`);
}
