import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

from models import ApartmentImageCategorization, HotelImageCategorization
from prompt import HOTEL_IMAGE_CATEGORIZATION_PROMPT
from prompt import APARTMENT_IMAGE_CATEGORIZATION_PROMPT

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


def encode_image_part(image_path: str) -> types.Part:
    """Encodes a local image file into a types.Part object for the API."""

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    return types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png",
    )


def categorize_hotel_image(
    image_path: str, client: genai.Client
) -> HotelImageCategorization:
    """Categorizes a hotel image into a specific category."""

    image_part = encode_image_part(image_path)
    contents = [HOTEL_IMAGE_CATEGORIZATION_PROMPT, image_part]

    response_config = types.GenerateContentConfig(
        temperature=0.0,  # No randomness, "deterministic" output
        response_mime_type="application/json",
        response_schema=HotelImageCategorization.model_json_schema(),
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config=response_config,
        contents=contents,
    )
    response_json = json.loads(response.text)

    hotel_image_categorization = HotelImageCategorization(
        category=response_json["category"],
        reason=response_json["reason"],
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=(
            response.usage_metadata.candidates_token_count
            + response.usage_metadata.thoughts_token_count
        ),
    )

    assert (
        response.usage_metadata.total_token_count
        == response.usage_metadata.prompt_token_count
        + response.usage_metadata.candidates_token_count
        + response.usage_metadata.thoughts_token_count
    )

    return hotel_image_categorization


def categorize_apartment_image(
    image_path: str, client: genai.Client
) -> ApartmentImageCategorization:
    """Categorizes an apartment image into a specific category."""

    image_part = encode_image_part(image_path)
    contents = [APARTMENT_IMAGE_CATEGORIZATION_PROMPT, image_part]

    response_config = types.GenerateContentConfig(
        temperature=0.0,  # No randomness, "deterministic" output
        response_mime_type="application/json",
        response_schema=ApartmentImageCategorization.model_json_schema(),
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config=response_config,
        contents=contents,
    )
    response_json = json.loads(response.text)

    apartment_image_categorization = ApartmentImageCategorization(
        category=response_json["category"],
        reason=response_json["reason"],
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count
        + response.usage_metadata.thoughts_token_count,
    )

    assert (
        response.usage_metadata.total_token_count
        == response.usage_metadata.prompt_token_count
        + response.usage_metadata.candidates_token_count
        + response.usage_metadata.thoughts_token_count
    )

    return apartment_image_categorization


def main():
    """Main function to categorize a list of images."""

    # Initialize the Gemini API client
    client = genai.Client(api_key=GOOGLE_API_KEY)

    image_paths = [
        os.path.join("images/apartments_sixtyfour", f)
        for f in os.listdir("images/apartments_sixtyfour")
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    categorizations = []
    for image_path in image_paths:
        categorization = categorize_apartment_image(image_path, client)
        categorizations.append(categorization)
        print(f"Image path: {image_path}")
        print(f"Categorization: {categorization}")
        print("-" * 100)
        print(f"INPUT TOKENS: {categorization.input_tokens}")
        print(f"OUTPUT TOKENS: {categorization.output_tokens}")

    print(f"TOTAL IMAGES: {len(categorizations)}")
    print(
        f"TOTAL INPUT TOKENS: {sum(categorization.input_tokens for categorization in categorizations)}"
    )
    print(
        f"TOTAL OUTPUT TOKENS: {sum(categorization.output_tokens for categorization in categorizations)}"
    )


if __name__ == "__main__":
    main()
