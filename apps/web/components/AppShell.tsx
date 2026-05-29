"use client";

import { Activity, BarChart3, LayoutDashboard, LogOut, Waves } from "lucide-react";
import clsx from "clsx";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { ApiError, api } from "@/lib/api";
import { useAuthStore } from "@/store/auth";

export function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { token, user, hydrated, hydrate, clearSession } = useAuthStore();

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    if (hydrated && !token) {
      router.replace("/login");
    }
  }, [hydrated, router, token]);

  useEffect(() => {
    if (!hydrated || !token) return;
    api.me(token).catch((err) => {
      if (err instanceof ApiError && err.status === 401) {
        clearSession();
        router.replace("/login");
      }
    });
  }, [clearSession, hydrated, router, token]);

  if (!hydrated || !token) {
    return <div className="min-h-screen bg-surface" />;
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-3">
            <Link href="/dashboard" className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-md bg-water text-white">
                <Waves size={22} />
              </span>
              <span>
                <span className="block text-base font-bold text-ink">AquaIQ</span>
                <span className="block text-xs text-slate-500">Coach Console</span>
              </span>
            </Link>

            <nav className="hidden items-center gap-1 md:flex">
              <NavLink href="/dashboard" active={pathname === "/dashboard"} icon={LayoutDashboard}>
                Dashboard
              </NavLink>
              <NavLink href="/dashboard/swim-analysis" active={pathname?.includes("/swim-analysis") ?? false} icon={BarChart3}>
                Motion Analysis
              </NavLink>
            </nav>

            <div className="flex items-center gap-3">
              <div className="hidden items-center gap-2 text-sm text-slate-600 sm:flex">
                <Activity size={16} className="text-mint" />
                <span>{user?.full_name}</span>
              </div>
              <button
                className="icon-button"
                type="button"
                title="Sign out"
                aria-label="Sign out"
                onClick={() => {
                  clearSession();
                  router.replace("/login");
                }}
              >
                <LogOut size={17} />
              </button>
            </div>
          </div>
          <nav className="flex gap-1 border-t border-slate-100 py-2 md:hidden">
            <NavLink href="/dashboard" active={pathname === "/dashboard"} icon={LayoutDashboard}>
              Dashboard
            </NavLink>
            <NavLink href="/dashboard/swim-analysis" active={pathname?.includes("/swim-analysis") ?? false} icon={BarChart3}>
              Motion Analysis
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">{children}</main>
    </div>
  );
}

function NavLink({
  href,
  active,
  icon: Icon,
  children
}: {
  href: string;
  active: boolean;
  icon: typeof LayoutDashboard;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className={clsx(
        "inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-semibold transition",
        active ? "bg-blue-50 text-water" : "text-slate-600 hover:bg-slate-50 hover:text-water"
      )}
    >
      <Icon size={16} />
      {children}
    </Link>
  );
}
