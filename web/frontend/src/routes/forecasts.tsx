import { createFileRoute } from '@tanstack/react-router'
import { ForecastsPage } from '@/features/forecasts/pages'

export const Route = createFileRoute('/forecasts')({
  component: ForecastsPage,
})
