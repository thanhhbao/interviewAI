import { createClient } from '@/lib/supabase/client';

// Same-origin by default: on Vercel the backend service is routed under /api.
// Set NEXT_PUBLIC_BACKEND_URL (e.g. http://localhost:8000) for local non-`vercel dev` runs.
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? '';

/**
 * Fetch wrapper for calls to the Python backend.
 * Automatically attaches `Authorization: Bearer <token>` from the current
 * Supabase session. The backend must verify this token with Supabase's JWKS.
 */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const supabase = createClient();
  const { data: { session } } = await supabase.auth.getSession();
  const token = session?.access_token;

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init.headers as Record<string, string> | undefined),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return fetch(`${BACKEND_URL}${path}`, { ...init, headers });
}
