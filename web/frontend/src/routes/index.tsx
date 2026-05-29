import { createFileRoute } from '@tanstack/react-router'
import { DashboardPage } from '@/features/dashboard/pages'

export const Route = createFileRoute('/')({
  component: DashboardPage,
})
