"use client";

import { create } from "zustand";
import type { User } from "@/lib/types";

type AuthState = {
  token: string | null;
  user: User | null;
  hydrated: boolean;
  hydrate: () => void;
  setSession: (token: string, user: User) => void;
  clearSession: () => void;
};

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  user: null,
  hydrated: false,
  hydrate: () => {
    if (typeof window === "undefined") {
      return;
    }
    const token = window.localStorage.getItem("aquaiq_token");
    const userRaw = window.localStorage.getItem("aquaiq_user");
    set({
      token,
      user: userRaw ? (JSON.parse(userRaw) as User) : null,
      hydrated: true
    });
  },
  setSession: (token, user) => {
    window.localStorage.setItem("aquaiq_token", token);
    window.localStorage.setItem("aquaiq_user", JSON.stringify(user));
    set({ token, user, hydrated: true });
  },
  clearSession: () => {
    window.localStorage.removeItem("aquaiq_token");
    window.localStorage.removeItem("aquaiq_user");
    set({ token: null, user: null, hydrated: true });
  }
}));
