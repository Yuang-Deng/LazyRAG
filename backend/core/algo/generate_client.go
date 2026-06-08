package algo

import (
	"context"
	"fmt"
	"strings"
	"time"

	"lazymind/core/common"
)

const generateTimeout = 10 * time.Minute
const llmGeneratePath = "/api/chat/llm_generate"

func GenerateSkill(ctx context.Context, req SkillGenerateRequest) (string, error) {
	return generate(ctx, llmGeneratePayload("skill", req.Content, req.UserInstruct, req.LLMConfig))
}

func GenerateMemory(ctx context.Context, req ManagedGenerateRequest) (string, error) {
	return generate(ctx, llmGeneratePayload("memory", req.Content, req.UserInstruct, req.LLMConfig))
}

func GenerateUserPreference(ctx context.Context, req ManagedGenerateRequest) (string, error) {
	return generate(ctx, llmGeneratePayload("user_preference", req.Content, req.UserInstruct, req.LLMConfig))
}

func GeneratePolish(ctx context.Context, req PolishGenerateRequest) (string, error) {
	return generate(ctx, llmGeneratePayload("polish", req.Content, req.UserInstruct, req.LLMConfig))
}

func generateURL(path string) string {
	return common.ChatServiceEndpoint() + path
}

func generate(ctx context.Context, req LLMGenerateRequest) (string, error) {
	url := generateURL(llmGeneratePath)
	var response map[string]any
	if err := common.ApiPost(ctx, url, req, nil, &response, generateTimeout); err != nil {
		return "", err
	}
	content := extractGeneratedContent(response)
	if strings.TrimSpace(content) == "" {
		return "", fmt.Errorf("generate endpoint returned empty content")
	}
	return content, nil
}

func llmGeneratePayload(taskType, content, userInstruct string, llmConfig map[string]any) LLMGenerateRequest {
	if llmConfig == nil {
		llmConfig = map[string]any{}
	}
	return LLMGenerateRequest{
		TaskType:     taskType,
		Content:      content,
		UserInstruct: strings.TrimSpace(userInstruct),
		LLMConfig:    llmConfig,
	}
}

func extractGeneratedContent(payload any) string {
	switch typed := payload.(type) {
	case map[string]any:
		if data, ok := typed["data"]; ok {
			if s := extractGeneratedContent(data); strings.TrimSpace(s) != "" {
				return strings.TrimSpace(s)
			}
		}
		for _, key := range []string{"content", "text", "result", "generated_content", "full_content"} {
			if value, ok := typed[key]; ok {
				if s := extractGeneratedContent(value); strings.TrimSpace(s) != "" {
					return strings.TrimSpace(s)
				}
			}
		}
	case string:
		return strings.TrimSpace(typed)
	}
	return ""
}
