import React from "react"

export function Input({ className = "", ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input className={`border rounded px-3 py-2 ${className}`} {...props} />
} 