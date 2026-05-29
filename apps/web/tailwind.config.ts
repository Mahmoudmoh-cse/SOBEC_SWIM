import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
    "./store/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#22313f",
        water: "#1a56a0",
        mint: "#0f6e56",
        coral: "#d85a30",
        violet: "#6f65d8",
        surface: "#f7f8fa"
      },
      boxShadow: {
        panel: "0 1px 2px rgba(24, 39, 75, 0.06), 0 8px 24px rgba(24, 39, 75, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
