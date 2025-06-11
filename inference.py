import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder
from sagemaker_inference.default_handler_service import DefaultHandlerService

class ResumeCoachHandler(DefaultHandlerService):
    def __init__(self):
        self.initialized = False
        self.model = None
        self.tokenizer = None

    def initialize(self, context):
        """Initialize the model and tokenizer."""
        if self.initialized:
            return

        # Load model and tokenizer
        model_path = os.path.join(context.model_dir, "llama-2-7b")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        self.initialized = True

    def inference(self, model_input, context):
        """Perform inference on the input data."""
        if not self.initialized:
            self.initialize(context)

        # Parse input
        input_data = json.loads(model_input[0].decode())
        prompt = input_data.get("prompt", "")
        max_length = input_data.get("max_length", 2048)
        temperature = input_data.get("temperature", 0.7)

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return [json.dumps({"response": response}).encode()]

    def handle(self, data, context):
        """Handle inference requests."""
        if data is None:
            return None

        model_input = self.input_fn(data, context)
        model_output = self.inference(model_input, context)
        return self.output_fn(model_output, context)

    def input_fn(self, input_data, context):
        """Deserialize and prepare the prediction input."""
        if context.request_content_type == content_types.JSON:
            return decoder.decode(input_data, content_types.JSON)
        raise ValueError(f"Unsupported content type: {context.request_content_type}")

    def output_fn(self, prediction_output, context):
        """Serialize and prepare the prediction output."""
        if context.accept_content_type == content_types.JSON:
            return prediction_output
        raise ValueError(f"Unsupported accept content type: {context.accept_content_type}")

_service = ResumeCoachHandler()

def handle(data, context):
    """Entry point for SageMaker inference."""
    return _service.handle(data, context) 