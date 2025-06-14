import React from "react"

export function Badge({ children, className = "", variant = "primary" }: { children: React.ReactNode; className?: string; variant?: string }) {
  const base = "inline-block px-2 py-1 rounded text-xs font-semibold";
  const variants: Record<string, string> = {
    primary: "bg-red-600 text-white",
    secondary: "bg-gray-200 text-gray-800",
    outline: "border border-red-600 text-red-600 bg-white",
  };
  return <span className={`${base} ${variants[variant] || ""} ${className}`}>{children}</span>;
} 