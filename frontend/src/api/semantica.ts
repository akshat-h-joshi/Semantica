export interface RecommendationItem {
  paper_id: string
  title: string
  score: number
  keywords: string[]
  abstract: string
  link: string
}

export interface RecommendResponse {
  query: string
  model: string
  results: RecommendationItem[]
}

export async function recommend(
  query: string,
  model: string,
  topK: number
): Promise<RecommendResponse> {
  const res = await fetch("http://127.0.0.1:8000/api/v1/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      model,
      top_k: topK
    })
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text)
  }

  return res.json()
}