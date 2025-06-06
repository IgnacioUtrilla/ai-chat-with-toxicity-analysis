"use server";

import { createStreamableValue } from "ai/rsc";

// Function to analyze text toxicity using Hugging Face model
export async function analyzeToxicity(text: string) {
  const stream = createStreamableValue();

  await (async () => {
    try {
      // Import HuggingFace inference dynamically to avoid server-side issues
      const { HfInference } = await import("@huggingface/inference");

      // Initialize the Hugging Face inference with API token from environment variable
      const hf = new HfInference(process.env.HUGGINGFACE_API_TOKEN);

      // Call the toxicity classifier model
      const result = await hf.textClassification({
        model: "s-nlp/roberta_toxicity_classifier",
        inputs: text,
      });

      // Update the stream with the result
      stream.update({
        toxicity: result,
        analyzed_text: text,
      });
    } catch (error) {
      // Handle errors
      stream.update({
        error: error instanceof Error ? error.message : "Unknown error occurred",
        analyzed_text: text,
      });
    } finally {
      stream.done();
    }
  })();

  return { streamData: stream.value };
}

