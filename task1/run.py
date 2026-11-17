"""LoRA fine-tuning and held-out evaluation for a four-intent JSON task."""

from __future__ import annotations

import csv
import json
import random
import re
import time
from pathlib import Path

import torch
import transformers
import peft
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "task1" / "output"
RESULTS = ROOT / "results"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 20261117
INTENTS = ("billing", "technical_support", "account_access", "order_status")
SYSTEM_TEXT = (
    "Classify the customer message into exactly one of these intents: "
    "billing, technical_support, account_access, order_status. "
    'Return only a JSON object with one key, "intent".'
)

TRAIN_TEMPLATES = {
    "billing": [
        "I was charged twice for {item}.",
        "Why is my {period} invoice higher than I expected?",
        "Please refund the {charge} on my account.",
        "How do I change the card used for {item}?",
        "I don't recognize a {amount} charge from {merchant}.",
        "Can you send me the receipt for {item}?",
        "The payment for {item} was billed to the wrong card.",
        "I need help understanding the tax on my {period} bill.",
        "Please stop the recurring payment for {item}.",
        "My refund for {item} has not appeared on the statement.",
        "The invoice says {amount}, but I expected a different total.",
        "Where can I see the payment history for {item}?",
    ],
    "technical_support": [
        "The {device} keeps {problem} when I use {app}.",
        "I get {error} whenever I try to {action}.",
        "After {event}, the {app} will not {action}.",
        "Can you help me fix {problem} on my {device}?",
        "The {app} stopped working after the latest update.",
        "I cannot get {device} to connect to {network}.",
        "The screen freezes every time I open {app}.",
        "How do I resolve {error} on {device}?",
        "The {action} feature is broken in the app.",
        "My {device} is running slowly when {event}.",
        "The app crashes before I can finish {action}.",
        "I need troubleshooting steps for {problem}.",
    ],
    "account_access": [
        "I can't sign in after {event}.",
        "How can I reset the password for my {account}?",
        "I no longer have access to the {contact} on my profile.",
        "The verification code for my login never arrives.",
        "Please help me unlock my {account}.",
        "I forgot which {contact} I used to register.",
        "Two-factor authentication is stopping me from logging in.",
        "My account says the password is incorrect, but it is not.",
        "How do I update the recovery {contact} for my account?",
        "I am locked out after too many sign-in attempts.",
        "I need to change the email address used to log in.",
        "The sign-in link for my account has expired.",
    ],
    "order_status": [
        "Where is my {item} order?",
        "Can you check the tracking status for {item}?",
        "When should the {item} shipment arrive?",
        "My order {order_id} has not moved for several days.",
        "Has my package left the warehouse yet?",
        "I need an update on the delivery of {item}.",
        "The tracking page has no new information about my parcel.",
        "What is the estimated arrival date for order {order_id}?",
        "Please tell me whether my {item} has shipped.",
        "The courier has not delivered my package.",
        "I want to know where the driver is with my delivery.",
        "Can I get the latest shipping update for {order_id}?",
    ],
}

SLOTS = {
    "item": ["monthly plan", "replacement charger", "annual subscription", "desk lamp"],
    "period": ["March", "April", "current", "annual"],
    "charge": ["renewal fee", "duplicate payment", "service charge", "late fee"],
    "amount": ["$18.40", "$42.00", "$7.95", "$103.20"],
    "merchant": ["Northwind", "the app store", "Acme", "my provider"],
    "device": ["phone", "laptop", "tablet", "router"],
    "problem": ["freezing", "restarting", "showing a blank screen", "losing connection"],
    "app": ["mobile app", "desktop client", "dashboard", "browser"],
    "error": ["error 502", "a timeout", "an access denied message", "a blank page"],
    "action": ["upload a photo", "open my settings", "submit the form", "sync my files"],
    "event": ["the update", "restarting", "changing networks", "installing the patch"],
    "network": ["the home network", "Wi-Fi", "the office network", "Bluetooth"],
    "account": ["customer account", "work account", "profile", "online account"],
    "contact": ["old email address", "phone number", "recovery email", "mobile number"],
    "order_id": ["A-1048", "B-2091", "C-7710", "D-3402"],
}

TEST_PHRASES = {
    "billing": [
        "There is a duplicate debit on my statement.",
        "I need an itemized copy of last month's bill.",
        "Please explain why my renewal cost went up.",
        "The card on file was billed after I cancelled.",
        "Can someone reverse the fee I was charged yesterday?",
        "My account was charged in euros instead of dollars.",
        "I want to update the payment details for my membership.",
        "The invoice total does not match the advertised price.",
        "I have not received the credit you promised.",
        "Where do I download a tax invoice?",
        "A subscription payment failed; how can I pay it?",
        "I was billed for a service I never ordered.",
        "Can I get a receipt for the transaction on Tuesday?",
        "Please remove the extra service fee from my bill.",
        "My bank shows two pending payments for one purchase.",
        "When will the cancellation refund be processed?",
        "I need to change the billing cycle on my plan.",
        "Why was sales tax added to this order?",
        "The latest invoice was sent to the wrong email.",
        "How can I dispute an unfamiliar transaction?",
    ],
    "technical_support": [
        "The sign-in page spins forever in Firefox.",
        "My tablet loses its connection whenever the app launches.",
        "I see a 503 message when I save my profile.",
        "The camera upload gets stuck at ninety percent.",
        "After the patch, notifications no longer appear.",
        "The desktop program closes whenever I export a report.",
        "My browser cannot load the settings screen.",
        "The device will not pair with my headphones.",
        "Search returns a blank screen in the mobile client.",
        "The latest version made the dashboard very slow.",
        "I cannot print from the application anymore.",
        "A network timeout appears when I try to sync.",
        "The app's audio playback is cutting out.",
        "How do I fix a frozen screen on my laptop?",
        "The form gives an error when I attach a document.",
        "My router drops the connection every few minutes.",
        "The menu disappears when I rotate my phone.",
        "I need help getting the software to install.",
        "Exported files are empty even though the preview is full.",
        "The page reloads repeatedly after I press submit.",
    ],
    "account_access": [
        "My password manager's saved credentials are rejected.",
        "I can't get past the verification step on my profile.",
        "The reset email is going to an address I no longer use.",
        "I need to recover my username.",
        "The one-time login code arrived after it expired.",
        "My account was temporarily locked; how do I regain access?",
        "I changed phones and lost my authentication app.",
        "Where can I replace the recovery phone number?",
        "The password reset link says it has already been used.",
        "I am unable to log into the member portal.",
        "Can you change the email I use for authentication?",
        "My account asks for a code sent to my old number.",
        "I forgot my login password.",
        "The system says my user ID does not exist.",
        "I was signed out and cannot authenticate again.",
        "Please unlock my profile after the failed attempts.",
        "I need to set up a new second-factor device.",
        "The confirmation message for account recovery never arrived.",
        "How do I regain access without my former email?",
        "My sign-in credentials stopped working this morning.",
    ],
    "order_status": [
        "The parcel tracking number has not updated since Monday.",
        "Could you tell me the delivery window for my purchase?",
        "My shipment is marked dispatched but has not arrived.",
        "I need the current location of package Z-515.",
        "Has the warehouse handed my order to the carrier?",
        "The estimated arrival date disappeared from tracking.",
        "Please check whether my order is out for delivery.",
        "The courier's page says they have not received the parcel.",
        "When will the replacement item be delivered?",
        "My package seems delayed; can you look up its status?",
        "I am waiting for order R-880 to ship.",
        "Can I get a tracking update for the headphones I bought?",
        "The delivery has been in transit for a week.",
        "Is my purchase still expected to arrive tomorrow?",
        "I need to know if the carrier attempted delivery.",
        "Where can I see the latest progress on my shipment?",
        "The tracking link says the package is still at the depot.",
        "Please confirm the expected arrival of parcel P-942.",
        "My order is packed; has it been sent yet?",
        "Can you find the courier status for my recent purchase?",
    ],
}

CHALLENGE_PHRASES = [
    ("I forgot my password and cannot access the tracking page for my package.", "account_access"),
    ("The shipment still has not arrived; please credit the delivery fee.", "order_status"),
    ("The tracking screen crashes every time I open it in the mobile app.", "technical_support"),
    ("I need an invoice for the order that is still marked as unshipped.", "billing"),
    ("My profile is locked, and I need to change the card saved for my plan.", "account_access"),
    ("The app cannot load my invoice after the latest release.", "technical_support"),
    ("My payment went through, but where is the parcel now?", "order_status"),
    ("I cannot sign in to download the receipt for last month's renewal.", "account_access"),
    ("The delivery charge on my statement looks incorrect.", "billing"),
    ("The website shows an error when I try to follow my shipment.", "technical_support"),
    ("I was charged for an order that is still waiting to be dispatched.", "order_status"),
    ("Can I get a refund for the second charge on my membership?", "billing"),
    ("My password reset code arrives, but the account page freezes afterward.", "technical_support"),
    ("I lost access to the email used for my order account.", "account_access"),
    ("The parcel is late and I want to know whether it has shipped.", "order_status"),
    ("I need help with the invoice because the billing page will not open.", "technical_support"),
    ("My subscription renewed, but I did not authorize that payment.", "billing"),
    ("I cannot log in to update the delivery address on my order.", "account_access"),
    ("The courier says delivered, but I need the shipment location checked.", "order_status"),
    ("An error appears when I submit the payment details for my purchase.", "technical_support"),
]


def build_training_examples() -> list[tuple[str, str]]:
    rng = random.Random(SEED)
    examples: list[tuple[str, str]] = []
    for intent, templates in TRAIN_TEMPLATES.items():
        for template in templates:
            fields = re.findall(r"{([^}]+)}", template)
            for offset in range(5):
                values = {
                    field: SLOTS[field][(offset + template_index) % len(SLOTS[field])]
                    for template_index, field in enumerate(fields)
                }
                examples.append((template.format(**values), intent))
    rng.shuffle(examples)
    return examples


def build_test_examples() -> list[tuple[str, str]]:
    return [(text, intent) for intent, prompts in TEST_PHRASES.items() for text in prompts]


def chat_prompt(tokenizer, message: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": f"Message: {message}"},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    return encoded["input_ids"] if hasattr(encoded, "get") else encoded


def make_batch(tokenizer, batch, device, *, training: bool):
    pad_id = tokenizer.pad_token_id
    inputs, labels = [], []
    for prompt_ids, target_ids in batch:
        inputs.append(prompt_ids + target_ids)
        labels.append([-100] * len(prompt_ids) + target_ids)
    width = max(map(len, inputs))
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    label_ids = torch.full((len(batch), width), -100, dtype=torch.long)
    for row, (ids, row_labels) in enumerate(zip(inputs, labels)):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        attention[row, : len(ids)] = 1
        if training:
            label_ids[row, : len(row_labels)] = torch.tensor(row_labels)
    return input_ids.to(device), attention.to(device), label_ids.to(device)


def generate_predictions(model, tokenizer, examples, device, dtype) -> list[dict]:
    model.eval()
    predictions = []
    batch_size = 8
    for start in range(0, len(examples), batch_size):
        subset = examples[start : start + batch_size]
        prompts = [chat_prompt(tokenizer, text) for text, _ in subset]
        width = max(map(len, prompts))
        ids = torch.full(
            (len(prompts), width), tokenizer.pad_token_id, dtype=torch.long
        )
        attention = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            ids[row, width - len(prompt) :] = torch.tensor(prompt)
            attention[row, width - len(prompt) :] = 1
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids.to(device),
                attention_mask=attention.to(device),
                max_new_tokens=28,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row, (text, expected) in enumerate(subset):
            output = tokenizer.decode(
                generated[row, width:], skip_special_tokens=True
            ).strip()
            predictions.append(
                {"message": text, "expected": expected, "output": output}
            )
    return predictions


def parse_json_object(output: str) -> dict | None:
    match = re.search(r"\{[^{}]*\}", output)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_intent(output: str) -> str | None:
    value = parse_json_object(output)
    if value is None or set(value) != {"intent"}:
        return None
    intent = value["intent"]
    return intent if intent in INTENTS else None


def metrics(predictions: list[dict]) -> dict:
    valid = [parse_intent(row["output"]) for row in predictions]
    correct = sum(pred == row["expected"] for pred, row in zip(valid, predictions))
    return {
        "accuracy": correct / len(predictions),
        "valid_json_rate": (
            sum(parse_json_object(row["output"]) is not None for row in predictions)
            / len(predictions)
        ),
        "schema_valid_rate": sum(pred is not None for pred in valid) / len(predictions),
        "correct": correct,
        "examples": len(predictions),
    }


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_examples = build_training_examples()
    test_examples = build_test_examples()
    if len(test_examples) < 50:
        raise RuntimeError("Held-out evaluation must contain at least 50 examples")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    base_predictions = generate_predictions(model, tokenizer, test_examples, device, dtype)
    base_metrics = metrics(base_predictions)
    model.config.use_cache = False
    model.train()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=2e-4,
        weight_decay=0.01,
    )
    encoded = []
    for text, intent in train_examples:
        prompt = chat_prompt(tokenizer, text)
        answer = tokenizer.encode(
            json.dumps({"intent": intent}, separators=(",", ":")),
            add_special_tokens=False,
        )
        answer.append(tokenizer.eos_token_id)
        encoded.append((prompt, answer))

    batch_size = 4
    accumulation = 2
    epochs = 3
    started = time.perf_counter()
    optimizer_steps = 0
    for epoch in range(epochs):
        random.Random(SEED + epoch).shuffle(encoded)
        optimizer.zero_grad(set_to_none=True)
        batches = [encoded[i : i + batch_size] for i in range(0, len(encoded), batch_size)]
        for batch_index, batch in enumerate(batches):
            input_ids, attention, labels = make_batch(tokenizer, batch, device, training=True)
            loss = model(
                input_ids=input_ids,
                attention_mask=attention,
                labels=labels,
            ).loss
            (loss / accumulation).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batches)
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
    else:
        peak_memory = None
    training_seconds = time.perf_counter() - started

    model.config.use_cache = True
    tuned_predictions = generate_predictions(model, tokenizer, test_examples, device, dtype)
    tuned_metrics = metrics(tuned_predictions)
    challenge_examples = CHALLENGE_PHRASES
    challenge_predictions = generate_predictions(model, tokenizer, challenge_examples, device, dtype)
    challenge_rows = [
        {
            "message": row["message"],
            "expected": row["expected"],
            "output": row["output"],
            "prediction": parse_intent(row["output"]),
            "correct": parse_intent(row["output"]) == row["expected"],
        }
        for row in challenge_predictions
    ]
    model.save_pretrained(OUT / "adapter")
    tokenizer.save_pretrained(OUT / "adapter")

    merged = []
    for base, tuned in zip(base_predictions, tuned_predictions):
        merged.append(
            {
                "message": base["message"],
                "expected": base["expected"],
                "base_output": base["output"],
                "base_prediction": parse_intent(base["output"]),
                "tuned_output": tuned["output"],
                "tuned_prediction": parse_intent(tuned["output"]),
            }
        )
    with (RESULTS / "task1_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    details = {
        "model": MODEL_ID,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "seed": SEED,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "lora": {
            "rank": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "dropout": 0.05,
            "learning_rate": 2e-4,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation,
            "optimizer_steps": optimizer_steps,
            "train_examples": len(train_examples),
        },
        "test_examples": len(test_examples),
        "training_seconds": training_seconds,
        "peak_cuda_allocated_gib": peak_memory,
        "base": base_metrics,
        "lora_finetuned": tuned_metrics,
    }
    (RESULTS / "task1_metrics.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS / "task1_challenges.json").write_text(
        json.dumps(challenge_rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
