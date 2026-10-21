package edu.kit.kastel.sdq.lissa.ratlr.embeddingcreator;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.internal.RetryUtils;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import edu.kit.kastel.sdq.lissa.ratlr.Configuration;
import edu.kit.kastel.sdq.lissa.ratlr.Environment;
import okhttp3.Credentials;

public class OllamaEmbeddingCreator extends CachedEmbeddingCreator {
    // Input limit of bge-m3; also roughly the limit of OpenAI's text-embedding-3 models used for the GPT-4o results
    private static final int DEFAULT_NUM_CTX = 8192;

    public OllamaEmbeddingCreator(Configuration.ModuleConfiguration configuration) {
        super(
                configuration.argumentAsString("model", "nomic-embed-text:v1.5"),
                1,
                String.valueOf(configuration.argumentAsInt("num_ctx", DEFAULT_NUM_CTX)));
    }

    @Override
    protected EmbeddingModel createEmbeddingModel(String model, String... params) {
        String host = Environment.getenvNonNull("OLLAMA_EMBEDDING_HOST");
        String user = Environment.getenv("OLLAMA_EMBEDDING_USER");
        String password = Environment.getenv("OLLAMA_EMBEDDING_PASSWORD");

        String authorization = null;
        if (user != null && password != null && !user.isEmpty() && !password.isEmpty()) {
            authorization = Credentials.basic(user, password);
        }
        return new OllamaEmbedEndpointModel(host, model, Integer.parseInt(params[0]), authorization);
    }

    /**
     * Uses Ollama's /api/embed instead of the legacy /api/embeddings endpoint of langchain4j 0.33.
     * The legacy endpoint fails for inputs longer than the model's batch size (2048 tokens by default),
     * while /api/embed truncates them to their first num_ctx tokens, as {@link CachedEmbeddingCreator} does for OpenAI.
     */
    private static final class OllamaEmbedEndpointModel implements EmbeddingModel {
        private static final Logger logger = LoggerFactory.getLogger(OllamaEmbedEndpointModel.class);
        private static final ObjectMapper MAPPER = new ObjectMapper();
        private static final Duration TIMEOUT = Duration.ofMinutes(5);
        private static final int MAX_ATTEMPTS = 3;

        private final HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(TIMEOUT)
                .build();
        private final URI endpoint;
        private final String model;
        private final int numCtx;
        private final String authorization;

        private OllamaEmbedEndpointModel(String host, String model, int numCtx, String authorization) {
            Objects.requireNonNull(host, "OLLAMA_EMBEDDING_HOST is not set");
            this.endpoint = URI.create(host.replaceAll("/+$", "") + "/api/embed");
            this.model = model;
            this.numCtx = numCtx;
            this.authorization = authorization;
        }

        @Override
        public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
            List<Embedding> embeddings = new ArrayList<>();
            for (TextSegment textSegment : textSegments) {
                embeddings.add(RetryUtils.withRetry(() -> requestEmbedding(textSegment.text()), MAX_ATTEMPTS));
            }
            return Response.from(embeddings);
        }

        private Embedding requestEmbedding(String text) throws IOException, InterruptedException {
            // num_batch must hold the whole input, otherwise Ollama truncates to the batch size instead of num_ctx
            Map<String, Object> body = Map.of(
                    "model",
                    model,
                    "input",
                    text,
                    "truncate",
                    true,
                    "options",
                    Map.of("num_ctx", numCtx, "num_batch", numCtx));
            HttpRequest.Builder request = HttpRequest.newBuilder(endpoint)
                    .timeout(TIMEOUT)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(body)));
            if (authorization != null) {
                request.header("Authorization", authorization);
            }

            HttpResponse<String> response = client.send(request.build(), HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new IllegalStateException(
                        "Ollama embedding request failed (" + response.statusCode() + "): " + response.body());
            }
            JsonNode json = MAPPER.readTree(response.body());
            JsonNode vector = json.path("embeddings").path(0);
            if (!vector.isArray() || vector.isEmpty()) {
                throw new IllegalStateException("Ollama returned no embedding: " + response.body());
            }
            if (json.path("prompt_eval_count").asInt() >= numCtx) {
                logger.info("Input truncated to {} tokens by Ollama", numCtx);
            }

            float[] embedding = new float[vector.size()];
            for (int i = 0; i < embedding.length; i++) {
                embedding[i] = vector.get(i).floatValue();
            }
            return Embedding.from(embedding);
        }
    }
}
