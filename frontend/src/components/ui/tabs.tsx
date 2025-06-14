"use client"
import * as TabsPrimitive from "@radix-ui/react-tabs"
import React from "react"

export function Tabs({ defaultValue, className = "", children }: any) {
  return (
    <TabsPrimitive.Root defaultValue={defaultValue} className={className}>
      {children}
    </TabsPrimitive.Root>
  )
}

export function TabsList({ className = "", children }: any) {
  return <TabsPrimitive.List className={className}>{children}</TabsPrimitive.List>
}

export function TabsTrigger({ value, className = "", children }: any) {
  return (
    <TabsPrimitive.Trigger value={value} className={className}>
      {children}
    </TabsPrimitive.Trigger>
  )
}

export function TabsContent({ value, className = "", children }: any) {
  return (
    <TabsPrimitive.Content value={value} className={className}>
      {children}
    </TabsPrimitive.Content>
  )
} 