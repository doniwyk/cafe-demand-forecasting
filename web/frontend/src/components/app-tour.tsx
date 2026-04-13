import { useEffect } from 'react'
import { useRouterState } from '@tanstack/react-router'
import { Button } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useTour, hasCompleted } from '@/contexts/tour-context'
import { CircleHelpIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export function AppTour() {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const { startTour, isTourActive } = useTour()
  const { t } = useTranslation()

  useEffect(() => {
    if (isTourActive) return
    if (!hasCompleted(pathname)) {
      const timer = setTimeout(() => {
        startTour(pathname)
      }, 800)
      return () => clearTimeout(timer)
    }
  }, [pathname, isTourActive, startTour])

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={() => startTour(pathname)}
          disabled={isTourActive}
        >
          <CircleHelpIcon className="size-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>
        <p>{t('tour.startTour')}</p>
      </TooltipContent>
    </Tooltip>
  )
}
