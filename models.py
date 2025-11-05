from pydantic import BaseModel, Field
from enum import Enum


class HotelImageCategory(str, Enum):
    GUEST_ROOM = "Guest Room"
    BATHROOM = "Bathroom"
    LOBBY_COMMON_AREA = "Lobby / Common Area"
    RESTAURANTS_BAR = "Restaurants / Bar"
    FACILITIES_AMENITIES = "Facilities (Amenities)"
    EXTERIOR_FACADE = "Exterior / Facade"
    MISCELLANEOUS_DETAILS = "Miscellaneous / Details"
    OTHER = "Other"
    NON_RELEVANT = "Non-Relevant"


class ApartmentImageCategory(str, Enum):
    BEDROOM = "Bedroom"
    BATHROOM = "Bathroom"
    KITCHEN_UTILITY = "Kitchen / Utility"
    LIVING_DINING = "Living / Dining"
    EXTERIOR_BUILDING = "Exterior / Building"
    VIEW_BALCONY = "View / Balcony"
    OTHER = "Other"
    NON_RELEVANT = "Non-Relevant"


class HotelImageCategorization(BaseModel):
    """The structured output model for Hotel properties."""

    category: HotelImageCategory = Field(
        ...,
        description="The specific image category relevant to Hotel amenities and spaces.",
    )
    reason: str = Field(
        ..., description="A single-sentence justification for the category choice."
    )
    input_tokens: int = Field(
        ..., description="The number of tokens used to encode the input image."
    )
    output_tokens: int = Field(
        ..., description="The number of tokens used to generate the categorization."
    )


class ApartmentImageCategorization(BaseModel):
    """The structured output model for Apartment properties."""

    category: ApartmentImageCategory = Field(
        ...,
        description="The specific image category relevant to Apartment amenities and spaces.",
    )
    reason: str = Field(
        ..., description="A single-sentence justification for the category choice."
    )
    input_tokens: int = Field(
        ..., description="The number of tokens used to encode the input image."
    )
    output_tokens: int = Field(
        ..., description="The number of tokens used to generate the categorization."
    )
