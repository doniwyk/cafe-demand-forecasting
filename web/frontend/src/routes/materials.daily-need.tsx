import { createFileRoute } from '@tanstack/react-router'
import { DailyNeedPage } from '@/features/materials/pages'

export const Route = createFileRoute('/materials/daily-need')({
  component: DailyNeedPage,
})
