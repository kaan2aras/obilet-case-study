import React from "react"

export function Button({ children, className = "", variant = "primary", ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: string }) {
  const base = "px-4 py-2 rounded font-medium transition-colors";
  const variants: Record<string, string> = {
    primary: "bg-red-600 text-white hover:bg-red-700",
    outline: "border border-red-600 text-red-600 bg-white hover:bg-red-50",
  };
  return (
    <button className={`${base} ${variants[variant] || ""} ${className}`} {...props}>
      {children}
    </button>
  )
} 