/**
 * remote-llm.ts - Remote LLM implementation using OpenAI-compatible APIs
 *
 * Provides generate, expandQuery, and rerank via a remote API (e.g. OpenRouter),
 * while delegating embed/embedBatch to a local LlamaCpp instance for fast CPU embeddings.
 *
 * Configuration via environment variables:
 *   QMD_REMOTE_LLM_URL     - API base URL (default: https://openrouter.ai/api/v1)
 *   QMD_REMOTE_LLM_API_KEY - API key (required)
 *   QMD_REMOTE_LLM_MODEL   - Model for generation/reranking (default: amazon/nova-micro-v1)
 */

import {
  LlamaCpp,
  type LLM,
  type EmbedOptions,
  type EmbeddingResult,
  type GenerateOptions,
  type GenerateResult,
  type ModelInfo,
  type Queryable,
  type QueryType,
  type RerankDocument,
  type RerankOptions,
  type RerankResult,
} from "./llm.js";

const DEFAULT_REMOTE_URL = "https://openrouter.ai/api/v1";
const DEFAULT_REMOTE_MODEL = "openai/gpt-4.1-nano";

export type RemoteLLMConfig = {
  /** API base URL (default: https://openrouter.ai/api/v1) */
  remoteUrl?: string;
  /** API key (required) */
  apiKey: string;
  /** Model to use for generation/reranking (default: anthropic/claude-sonnet-4) */
  model?: string;
  /** Optional embed model URI to pass to local LlamaCpp instance */
  embedModel?: string;
};

/**
 * LLM implementation that offloads text generation and reranking to a remote
 * OpenAI-compatible API, while keeping embeddings local via LlamaCpp.
 */
export class RemoteLLM implements LLM {
  private readonly localLlm: LlamaCpp;
  private readonly remoteUrl: string;
  private readonly apiKey: string;
  private readonly model: string;

  constructor(config: RemoteLLMConfig) {
    this.remoteUrl = (config.remoteUrl ?? DEFAULT_REMOTE_URL).replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.model = config.model ?? DEFAULT_REMOTE_MODEL;

    // Local LlamaCpp used only for embeddings
    this.localLlm = new LlamaCpp({
      embedModel: config.embedModel,
      inactivityTimeoutMs: 5 * 60 * 1000,
      disposeModelsOnInactivity: true,
    });
  }

  // ==========================================================================
  // Embeddings — delegated to local LlamaCpp
  // ==========================================================================

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.localLlm.embed(text, options);
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.localLlm.embedBatch(texts);
  }

  // ==========================================================================
  // Remote generation
  // ==========================================================================

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    let response: string | null = null;
    try {
      response = await this._chatCompletion([
        { role: "user", content: prompt },
      ], {
        maxTokens: options?.maxTokens ?? 1024,
        temperature: options?.temperature ?? 0.7,
      });
    } catch (error) {
      console.error("RemoteLLM generate failed:", error);
    }

    if (!response) return null;

    return {
      text: response,
      model: this.model,
      done: true,
    };
  }

  // ==========================================================================
  // Query expansion — remote, parses lex:/vec:/hyde: format
  // ==========================================================================

  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean; intent?: string } = {}
  ): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;

    const systemPrompt = `You are a search query expansion assistant. Given a user query, expand it into multiple search variants for different backends.

Output ONLY lines in this exact format (no other text):
lex: <keyword search variant>
vec: <semantic search variant>
hyde: <hypothetical document excerpt that would answer the query>

Rules:
- Each line must start with exactly "lex: ", "vec: ", or "hyde: "
- Include all three types
- Keep variants relevant to the original query
- The hyde variant should be 1-2 sentences of hypothetical content`;

    const userMessage = options.intent
      ? `Expand this search query: ${query}\nQuery intent: ${options.intent}`
      : `Expand this search query: ${query}`;

    let responseText: string | null = null;
    try {
      responseText = await this._chatCompletion([
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage },
      ], { maxTokens: 400, temperature: 0.7 });
    } catch (error) {
      console.error("RemoteLLM expandQuery failed:", error);
    }

    if (!responseText) {
      return this._expandQueryFallback(query, includeLexical);
    }

    const queryLower = query.toLowerCase();
    const queryTerms = queryLower.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);

    const hasQueryTerm = (text: string): boolean => {
      const lower = text.toLowerCase();
      if (queryTerms.length === 0) return true;
      return queryTerms.some(term => lower.includes(term));
    };

    const lines = responseText.trim().split("\n");
    const queryables: Queryable[] = lines.map(line => {
      const colonIdx = line.indexOf(":");
      if (colonIdx === -1) return null;
      const type = line.slice(0, colonIdx).trim();
      if (type !== "lex" && type !== "vec" && type !== "hyde") return null;
      const text = line.slice(colonIdx + 1).trim();
      if (!text || !hasQueryTerm(text)) return null;
      return { type: type as QueryType, text };
    }).filter((q): q is Queryable => q !== null);

    const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== "lex");
    if (filtered.length > 0) return filtered;

    return this._expandQueryFallback(query, includeLexical);
  }

  private _expandQueryFallback(query: string, includeLexical: boolean): Queryable[] {
    const fallback: Queryable[] = [
      { type: "hyde", text: `Information about ${query}` },
      { type: "lex", text: query },
      { type: "vec", text: query },
    ];
    return includeLexical ? fallback : fallback.filter(q => q.type !== "lex");
  }

  // ==========================================================================
  // Reranking — prompt-based scoring via remote LLM
  // ==========================================================================

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options: RerankOptions = {}
  ): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: this.model };
    }

    const docList = documents
      .map((doc, i) => `Document ${i}:\n${doc.text.slice(0, 800)}`)
      .join("\n\n");

    const systemPrompt = `You are a document relevance scorer. Given a query and a list of documents, score each document's relevance to the query on a scale from 0 to 10.

Return ONLY a JSON array in this exact format (no other text):
[{"index": 0, "score": 8.5}, {"index": 1, "score": 3.2}, ...]

Include an entry for every document index provided. Scores should be floats between 0.0 and 10.0.`;

    const userMessage = `Query: ${query}\n\n${docList}`;

    let scores: { index: number; score: number }[] = [];
    try {
      const responseText = await this._chatCompletion([
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage },
      ], { maxTokens: 200, temperature: 0.1 });

      if (responseText) {
        // Extract JSON array from response (model may include preamble)
        const jsonMatch = responseText.match(/\[[\s\S]*\]/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]) as { index: number; score: number }[];
          scores = parsed.filter(
            item => typeof item.index === "number" && typeof item.score === "number"
          );
        }
      }
    } catch (error) {
      console.error("RemoteLLM rerank failed:", error);
    }

    // Build score map, defaulting to 0 for missing entries
    // Normalize from 0-10 to 0-1 to match LlamaCpp cosine-similarity scale
    const scoreMap = new Map(scores.map(s => [s.index, s.score / 10]));

    const results = documents
      .map((doc, i) => ({
        file: doc.file,
        score: scoreMap.get(i) ?? 0,
        index: i,
      }))
      .sort((a, b) => b.score - a.score);

    return { results, model: this.model };
  }

  // ==========================================================================
  // Model info
  // ==========================================================================

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  // ==========================================================================
  // Lifecycle
  // ==========================================================================

  async dispose(): Promise<void> {
    await this.localLlm.dispose();
  }

  // ==========================================================================
  // Internal helpers
  // ==========================================================================

  private async _chatCompletion(
    messages: { role: string; content: string }[],
    options: { maxTokens?: number; temperature?: number } = {}
  ): Promise<string | null> {
    const url = `${this.remoteUrl}/chat/completions`;

    const body = JSON.stringify({
      model: this.model,
      messages,
      max_tokens: options.maxTokens ?? 1024,
      temperature: options.temperature ?? 0.7,
    });

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`,
      },
      body,
      signal: AbortSignal.timeout(30_000),
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "(unreadable)");
      throw new Error(`RemoteLLM API error ${response.status}: ${errorText}`);
    }

    const data = await response.json() as {
      choices?: Array<{ message?: { content?: string } }>;
    };

    return data.choices?.[0]?.message?.content ?? null;
  }
}
