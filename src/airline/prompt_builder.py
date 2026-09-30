"""Build prompts for airline domain using the original RuleArena format."""

from pathlib import Path

_AIRLINE_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "rulearena" / "airline"

SYSTEM_PROMPT = (
    "You are a helpful assistant at American Airlines. You are given the "
    "information of a passenger, his / her items, his / her special needs, "
    "and the policies of American Airlines. You should compute the total cost "
    "(including the flight ticket fee, checked bag fees, cost of special needs) "
    "according to the policies for the passenger."
)

PROMPT_TEMPLATE = """
The policies of American Airlines are as follows:

{reference_rules}
{example_prompt}
{question_prompt} Compute the total cost for him step by step (don't omit any bag) and end your response with "The total cost is $xxx." (xxx is a number)
Your response:
"""

# Official one-shot example from RuleArena auto_test.py
ONE_SHOT_EXAMPLE = """
Here is an example for you to follow:
<example>
John is a Main Plus Class passenger flying from Orlando to Lima with the following items:
1. A backpack: 21 x 11 x 7 inches, 9 lbs;
2. A luggage box: 51 x 24 x 27 inches, 65 lbs;
3. A backpack: 46 x 29 x 24 inches, 85 lbs;
4. A luggage box: 35 x 12 x 11 inches, 44 lbs;
5. A luggage box: 30 x 11 x 10 inches, 33 lbs;
John's flight ticket is $448. Compute the total cost for him step by step and end your response with "The total cost is $xxx." (xxx is a number)
Your response:
To calculate the total cost for John, we need to consider the flight ticket cost, checked bag fees, and any overweight or oversize fees according to American Airlines' policies.

Flight Ticket Cost:
- John's flight ticket: $448

Carry-on Allowance:
- John is allowed 1 carry-on bag and 1 personal item.
- Item 1 (Backpack):
  - Dimensions: 21 x 11 x 7 inches
  - Weight: 9 lbs
  - This fits within the carry-on size limit of 22 x 14 x 9 inches.
  - Fee: $0 (Allowed as carry-on bag)
- John does not have any items that fit the personal item dimensions (18 x 14 x 8 inches). So, no personal item is carried.

Checked Bags:
- Items to be checked: Items 2, 3, 4, and 5
- John is a Main Plus passenger, which includes 1 extra free checked bag in addition to the Main Cabin allowance, for a total of 2 free checked bags.
- Checked Bag Fees:
  - First Bag: $0 (free)
  - Second Bag: $0 (free)
  - Third Bag: $200
  - Fourth Bag: $200

Fees for Each Checked Bag:

1. Item 2 (Luggage box):
   - Dimensions: 51 x 24 x 27 inches
     - Total dimensions: 51 + 24 + 27 = 102 inches
     - Over the standard size limit of 62 inches.
   - Weight: 65 lbs
     - Over the standard weight limit of 50 lbs but under 70 lbs.
   - Checking Fee:
     - For the first checked bag, the checking fee is $0.
   - Oversize Fee:
     - For dimensions over 65 inches up to 115 inches between the U.S. and South America, the fee is $150.
   - Overweight Fee:
     - For weights over 53 lbs up to 70 lbs, the fee is $100.
   - The higher of oversize and overweight fee should apply.
   - Total Fee for Item 2: $0 (checking) + $150 (oversize) = $150
2. Item 3 (Backpack):
   - Dimensions: 46 x 29 x 24 inches
     - Total dimensions: 46 + 29 + 24 = 99 inches
     - Over the standard size limit of 62 inches.
   - Weight: 85 lbs
     - Over the standard weight limit of 50 lbs and over 70 lbs but under 100 lbs.
   - Checking Fee:
     - For the second checked bag, the checking fee is $0.
   - Oversize Fee:
     - For dimensions over 65 inches up to 115 inches between the U.S. and South America, the fee is $150.
   - Overweight Fee:
     - For weights over 70 lbs up to 100 lbs, the fee is $200.
   - The higher of oversize and overweight fee should apply.
   - Total Fee for Item 3: $0 (checking) + $200 (overweight) = $200
3. Item 4 (Luggage box):
   - Dimensions: 35 x 12 x 11 inches
     - Total dimensions: 35 + 12 + 11 = 58 inches
     - Within the standard size limit of 62 inches.
   - Weight: 44 lbs
     - Within the standard weight limit of 50 lbs.
   - Checking Fee:
     - For the third checked bag, the checking fee is $200.
   - Total Fee for Item 4: $200 (checking) + $0 (No overweight or oversize fees) = $200
4. Item 5 (Luggage box):
   - Dimensions: 30 x 11 x 10 inches
     - Total dimensions: 30 + 11 + 10 = 51 inches
     - Within the standard size limit of 62 inches.
   - Weight: 33 lbs
     - Within the standard weight limit of 50 lbs.
   - Checking Fee:
     - For the fourth checked bag, the checking fee is $200.
   - Total Fee for Item 5: $200 (checking) + $0 (No overweight or oversize fees) = $200
Summary of Baggage Fees:
  - Item 2: $200
  - Item 3: $150
  - Item 4: $200
  - Item 5: $200
Total Baggage Fees: $200 (Item 2) + $150 (Item 3) + $200 (Item 4) + $200 (Item 5) = $750
Total Cost:
- Flight Ticket: $448
- Total Baggage Fees: $750
- The total cost is $1,198.
</example>
"""


def load_reference_rules(textual: bool = False) -> str:
    fname = "reference_rules_textual.txt" if textual else "reference_rules.txt"
    return (_AIRLINE_DIR / fname).read_text()


_CACHED_RULES = {}


def get_reference_rules(textual: bool = False) -> str:
    if textual not in _CACHED_RULES:
        _CACHED_RULES[textual] = load_reference_rules(textual)
    return _CACHED_RULES[textual]


def build_airline_prompt(question_prompt: str, textual: bool = False,
                         prompt_mode: str = "zero_shot") -> str:
    """Build the full user prompt for an airline problem.

    Args:
        prompt_mode: 'zero_shot' or 'one_shot'.
    """
    rules = get_reference_rules(textual)
    example = ONE_SHOT_EXAMPLE if prompt_mode == "one_shot" else ""
    return PROMPT_TEMPLATE.format(
        reference_rules=rules,
        example_prompt=example,
        question_prompt=question_prompt,
    )


def build_airline_messages(question_prompt: str, textual: bool = False,
                           prompt_mode: str = "zero_shot") -> list[dict]:
    """Build the full messages list for an airline problem."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_airline_prompt(
            question_prompt, textual, prompt_mode)},
    ]


def rebuild_question_prompt_from_original(original_prompt: str,
                                          original_info: dict,
                                          cf_info: dict) -> str:
    """Rebuild a question prompt by applying counterfactual changes to the original.

    Instead of reconstructing from scratch (which loses name/cities),
    we take the original prompt text and only change what's different.
    """
    prompt = original_prompt

    # Handle customer class change
    if cf_info["customer_class"] != original_info["customer_class"]:
        prompt = prompt.replace(
            f'{original_info["customer_class"]} Class',
            f'{cf_info["customer_class"]} Class')

    # Handle base price change
    if cf_info["base_price"] != original_info["base_price"]:
        prompt = prompt.replace(
            f'${original_info["base_price"]}.',
            f'${cf_info["base_price"]}.')

    # Handle bag changes
    for i, (orig_bag, cf_bag) in enumerate(
            zip(original_info["bag_list"], cf_info["bag_list"])):
        od = orig_bag["size"]
        cd = cf_bag["size"]
        ow = orig_bag["weight"]
        cw = cf_bag["weight"]

        if od != cd or ow != cw:
            old_str = (f'{od[0]} x {od[1]} x {od[2]} inches, {ow} lbs')
            new_str = (f'{cd[0]} x {cd[1]} x {cd[2]} inches, {cw} lbs')
            prompt = prompt.replace(old_str, new_str, 1)

    return prompt


def rebuild_question_prompt(info_dict: dict) -> str:
    """Fallback: reconstruct question prompt entirely from info_dict.

    Used only when the original prompt text is not available.
    """
    name = info_dict.get("name", "the passenger")
    bag_list = info_dict["bag_list"]
    customer_class = info_dict["customer_class"]
    departure = info_dict.get("departure", "the departure city")
    destination = info_dict.get("destination", "the destination city")

    lines = [f"{name} is a {customer_class} Class passenger flying from "
             f"{departure} to {destination} with the following items:"]

    for i, bag in enumerate(bag_list):
        dims = bag["size"]
        w = bag["weight"]
        bname = bag.get("name", "luggage box")
        lines.append(f"{i+1}. A {bname}: {dims[0]} x {dims[1]} x {dims[2]} inches, {w} lbs;")

    lines.append(f"\n{name}'s flight ticket is ${info_dict['base_price']}.")
    return "\n".join(lines)
