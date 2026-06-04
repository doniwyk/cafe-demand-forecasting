import { createFileRoute } from '@tanstack/react-router'
import { DashboardPage } from '@/features/analytics/pages/dashboard'

export const Route = createFileRoute('/')({
  component: DashboardPage,
})
