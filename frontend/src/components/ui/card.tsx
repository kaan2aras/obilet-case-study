import React from "react"

export function Card({ children, className = "" }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={`bg-white rounded-lg shadow ${className}`}>{children}</div>
}

export function CardContent({ children, className = "" }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={`p-4 ${className}`}>{children}</div>
} 