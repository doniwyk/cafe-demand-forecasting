import { createContext, useContext, useState, useMemo, type ReactNode } from 'react'

export const MODEL_TYPE = 'xgboost' as const
export type ModelType = typeof MODEL_TYPE

export const MODEL_LABEL = 'XGBoost'

interface ModelContextValue {
  modelType: ModelType
}

const ModelContext = createContext<ModelContextValue>({
  modelType: 'xgboost',
})

export function ModelProvider({ children }: { children: ReactNode }) {
  const value = useMemo(() => ({ modelType: MODEL_TYPE }), [])
  return (
    <ModelContext value={value}>
      {children}
    </ModelContext>
  )
}

export function useModelType() {
  return useContext(ModelContext)
}
