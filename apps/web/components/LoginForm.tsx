"use client";

import { LogIn, Waves } from "lucide-react";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth";

export function LoginForm() {
  const router = useRouter();
  const setSession = useAuthStore((state) => state.setSession);
  const [email, setEmail] = useState("coach@aquaiq.local");
  const [password, setPassword] = useState("AquaIQ123!");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const session = await api.login(email, password);
      setSession(session.access_token, session.user);
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4 py-10">
      <form onSubmit={onSubmit} className="panel w-full max-w-sm p-6">
        <div className="mb-8 flex items-center gap-3">
          <span className="flex h-11 w-11 items-center justify-center rounded-md bg-water text-white">
            <Waves size={24} />
          </span>
          <div>
            <h1 className="text-xl font-bold text-ink">AquaIQ</h1>
            <p className="text-sm text-slate-500">Coach Console</p>
          </div>
        </div>

        <div className="space-y-4">
          <label className="block">
            <span className="label">Email</span>
            <input className="field mt-1" value={email} onChange={(event) => setEmail(event.target.value)} />
          </label>
          <label className="block">
            <span className="label">Password</span>
            <input
              className="field mt-1"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
        </div>

        {error ? <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}

        <button className="primary-button mt-6 w-full" type="submit" disabled={loading}>
          <LogIn size={17} />
          {loading ? "Signing in" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
