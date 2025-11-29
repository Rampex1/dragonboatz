import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpStatusOptimal, lpSum, PULP_CBC_CMD


@dataclass
class Person:
    name: str
    gender: str
    weight: float
    side: str


@dataclass
class Boat:
    size: int
    gender: str  # "Open" (only males), "Women" (only females), "Mixed" (half male, half female)


def parse_csv(filepath: str) -> List[Person]:
    people = []
    try:
        with open(filepath, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                person = Person(
                    name=row["Name"],
                    gender=row["Gender"],
                    side=row["Side"],
                    weight=float(row["Weight"])
                )
                people.append(person)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please ensure it exists.")
        return []
    except KeyError as e:
        print(f"Error: Missing column in CSV file: {e}. Required columns are 'Name', 'Gender', 'Side', and 'Weight'.")
        return []
    except ValueError as e:
        print(f"Error: Could not convert a value to a number. Details: {e}")
        return []
    return people


def build_boat(people: List[Person], boats: List[Boat]) -> Dict[
    str, Dict[int, Tuple[List[Person], List[Person]]]]:
    """
    Use linear programming to assign people to multiple boats across two halves with optimal weight distribution.
    Second half requires substitutions - can only replace people, not rebuild entire boats.

    Args:
        people: List of available people
        boats: List of boats with size and gender requirements

    Returns:
        Dictionary with keys 'first_half' and 'second_half', each mapping boat index to (left_side, right_side) tuple

    Constraints:
    - All original constraints apply to both halves
    - Second half: only substitutions allowed (remove x people, add x people who weren't on any boat in first half)
    - Minimize the maximum weight difference across all boats in both halves
    """
    n_people = len(people)
    n_boats = len(boats)
    total_boat_capacity = sum(boat.size for boat in boats)

    # Validate boat sizes are even
    for i, boat in enumerate(boats):
        if boat.size % 2 != 0:
            raise ValueError(f"Boat {i} has odd size {boat.size}. All boats must have even sizes for left/right split.")

    # Validate we have enough people
    if n_people < total_boat_capacity:
        raise ValueError(f"Not enough people ({n_people}) for total boat capacity ({total_boat_capacity})")

    # Create the optimization problem
    prob = LpProblem("Two_Half_Boat_Selection", LpMinimize)

    # Decision variables for first half: assign_first[person][boat][side]
    assign_first = {}
    for p in range(n_people):
        assign_first[p] = {}
        for b in range(n_boats):
            assign_first[p][b] = {
                'left': LpVariable(f"assign_first_{p}_{b}_left", cat='Binary'),
                'right': LpVariable(f"assign_first_{p}_{b}_right", cat='Binary')
            }

    # Decision variables for second half: assign_second[person][boat][side]
    assign_second = {}
    for p in range(n_people):
        assign_second[p] = {}
        for b in range(n_boats):
            assign_second[p][b] = {
                'left': LpVariable(f"assign_second_{p}_{b}_left", cat='Binary'),
                'right': LpVariable(f"assign_second_{p}_{b}_right", cat='Binary')
            }

    # Variable for maximum weight difference across both halves
    max_weight_diff = LpVariable("max_weight_diff", lowBound=0)

    # FIRST HALF CONSTRAINTS
    # Each person can be assigned to at most one boat and one side in first half
    for p in range(n_people):
        total_assignments = []
        for b in range(n_boats):
            total_assignments.extend([assign_first[p][b]['left'], assign_first[p][b]['right']])
        prob += lpSum(total_assignments) <= 1

    # Each boat must be filled to exactly its size in first half
    for b in range(n_boats):
        boat_total = []
        for p in range(n_people):
            boat_total.extend([assign_first[p][b]['left'], assign_first[p][b]['right']])
        prob += lpSum(boat_total) == boats[b].size

    # Equal numbers on left and right sides of each boat in first half
    for b in range(n_boats):
        left_count = lpSum(assign_first[p][b]['left'] for p in range(n_people))
        right_count = lpSum(assign_first[p][b]['right'] for p in range(n_people))
        prob += left_count == boats[b].size // 2
        prob += right_count == boats[b].size // 2

    # Gender constraints for first half
    for b in range(n_boats):
        boat = boats[b]
        if boat.gender == "Open":
            for p in range(n_people):
                if people[p].gender == 'F':
                    prob += assign_first[p][b]['left'] == 0
                    prob += assign_first[p][b]['right'] == 0
        elif boat.gender == "Women":
            for p in range(n_people):
                if people[p].gender == 'M':
                    prob += assign_first[p][b]['left'] == 0
                    prob += assign_first[p][b]['right'] == 0
        elif boat.gender == "Mixed":
            male_count = lpSum(assign_first[p][b]['left'] + assign_first[p][b]['right']
                               for p in range(n_people) if people[p].gender == 'M')
            female_count = lpSum(assign_first[p][b]['left'] + assign_first[p][b]['right']
                                 for p in range(n_people) if people[p].gender == 'F')
            prob += male_count == boat.size // 2
            prob += female_count == boat.size // 2

    # Handedness constraints for first half
    for p in range(n_people):
        for b in range(n_boats):
            if people[p].side == 'L':
                prob += assign_first[p][b]['right'] == 0
            elif people[p].side == 'R':
                prob += assign_first[p][b]['left'] == 0

    # SECOND HALF CONSTRAINTS
    # Each person can be assigned to at most one boat and one side in second half
    for p in range(n_people):
        total_assignments = []
        for b in range(n_boats):
            total_assignments.extend([assign_second[p][b]['left'], assign_second[p][b]['right']])
        prob += lpSum(total_assignments) <= 1

    # Each boat must be filled to exactly its size in second half
    for b in range(n_boats):
        boat_total = []
        for p in range(n_people):
            boat_total.extend([assign_second[p][b]['left'], assign_second[p][b]['right']])
        prob += lpSum(boat_total) == boats[b].size

    # Equal numbers on left and right sides of each boat in second half
    for b in range(n_boats):
        left_count = lpSum(assign_second[p][b]['left'] for p in range(n_people))
        right_count = lpSum(assign_second[p][b]['right'] for p in range(n_people))
        prob += left_count == boats[b].size // 2
        prob += right_count == boats[b].size // 2

    # Gender constraints for second half
    for b in range(n_boats):
        boat = boats[b]
        if boat.gender == "Open":
            for p in range(n_people):
                if people[p].gender == 'F':
                    prob += assign_second[p][b]['left'] == 0
                    prob += assign_second[p][b]['right'] == 0
        elif boat.gender == "Women":
            for p in range(n_people):
                if people[p].gender == 'M':
                    prob += assign_second[p][b]['left'] == 0
                    prob += assign_second[p][b]['right'] == 0
        elif boat.gender == "Mixed":
            male_count = lpSum(assign_second[p][b]['left'] + assign_second[p][b]['right']
                               for p in range(n_people) if people[p].gender == 'M')
            female_count = lpSum(assign_second[p][b]['left'] + assign_second[p][b]['right']
                                 for p in range(n_people) if people[p].gender == 'F')
            prob += male_count == boat.size // 2
            prob += female_count == boat.size // 2

    # Handedness constraints for second half
    for p in range(n_people):
        for b in range(n_boats):
            if people[p].side == 'L':
                prob += assign_second[p][b]['right'] == 0
            elif people[p].side == 'R':
                prob += assign_second[p][b]['left'] == 0

    # SUBSTITUTION CONSTRAINT: People in second half boats can only be:
    # 1) People who were on the same boat in first half (continuing)
    # 2) People who were NOT on any boat in first half (substitutes)
    for p in range(n_people):
        # A variable to represent if a person was on any boat in the first half
        on_any_boat_first = lpSum(assign_first[p][b]['left'] + assign_first[p][b]['right'] for b in range(n_boats))

        # A variable to represent if a person is on any boat in the second half
        on_any_boat_second = lpSum(assign_second[p][b]['left'] + assign_second[p][b]['right'] for b in range(n_boats))

        # We need a new binary variable to represent if the person is a substitute (i.e., on boat in second half, but not first)
        is_substitute = LpVariable(f"is_substitute_{p}", cat='Binary')
        prob += is_substitute <= on_any_boat_second
        prob += is_substitute <= 1 - on_any_boat_first
        prob += is_substitute >= on_any_boat_second + (1 - on_any_boat_first) - 1

        # The number of people assigned to a boat in the second half must be less than or equal to the number
        # of people assigned to that same boat in the first half plus the total number of substitutes.
        # This constraint is actually much simpler than the original one. We only need to ensure the group of substitutes
        # and the group of first half participants are disjoint. The total number of people on the second half boats
        # must equal the total capacity.

        # Let's ensure a person on a second-half boat is either a continuer from the same boat or a new substitute.
        for b in range(n_boats):
            on_boat_b_first = assign_first[p][b]['left'] + assign_first[p][b]['right']
            on_boat_b_second = assign_second[p][b]['left'] + assign_second[p][b]['right']
            prob += on_boat_b_second <= on_boat_b_first + (1 - on_any_boat_first)

    # Weight difference constraints for both halves
    # First half boats
    for b in range(n_boats):
        left_weight = lpSum(people[p].weight * assign_first[p][b]['left'] for p in range(n_people))
        right_weight = lpSum(people[p].weight * assign_first[p][b]['right'] for p in range(n_people))
        prob += max_weight_diff >= left_weight - right_weight
        prob += max_weight_diff >= right_weight - left_weight

    # Second half boats
    for b in range(n_boats):
        left_weight = lpSum(people[p].weight * assign_second[p][b]['left'] for p in range(n_people))
        right_weight = lpSum(people[p].weight * assign_second[p][b]['right'] for p in range(n_people))
        prob += max_weight_diff >= left_weight - right_weight
        prob += max_weight_diff >= right_weight - left_weight

    # Objective: Minimize maximum weight difference across all boats in both halves
    prob += max_weight_diff

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))

    # Check if solution is optimal
    if prob.status != LpStatusOptimal:
        raise ValueError(f"No optimal solution found. Status: {LpStatus[prob.status]}")

    # Extract the solution
    result = {
        'first_half': {},
        'second_half': {}
    }

    # Extract first half
    for b in range(n_boats):
        left_people = []
        right_people = []
        for p in range(n_people):
            if assign_first[p][b]['left'].varValue == 1:
                left_people.append(people[p])
            elif assign_first[p][b]['right'].varValue == 1:
                right_people.append(people[p])
        result['first_half'][b] = (left_people, right_people)

    # Extract second half
    for b in range(n_boats):
        left_people = []
        right_people = []
        for p in range(n_people):
            if assign_second[p][b]['left'].varValue == 1:
                left_people.append(people[p])
            elif assign_second[p][b]['right'].varValue == 1:
                right_people.append(people[p])
        result['second_half'][b] = (left_people, right_people)

    return result


# Helper function to print results
def print_half(half_name, assignments, boats, people):
    print(f"\n--- {half_name.upper().replace('_', ' ')} ASSIGNMENTS ---")
    for boat_idx, (left, right) in assignments.items():
        print(f"\n=== BOAT {boat_idx + 1} ({boats[boat_idx].gender}, size {boats[boat_idx].size}) ===")
        print("Left side:")
        for p in left:
            print(f"  - {p.name} ({p.gender}, {p.weight:.1f}kg, {p.side})")
        print("\nRight side:")
        for p in right:
            print(f"  - {p.name} ({p.gender}, {p.weight:.1f}kg, {p.side})")

        left_weight = sum(p.weight for p in left)
        right_weight = sum(p.weight for p in right)
        weight_diff = abs(left_weight - right_weight)

        print(f"\nWeight difference: {weight_diff:.2f}")
        print(f"Left side weight: {left_weight:.2f}")
        print(f"Right side weight: {right_weight:.2f}")

        # Verify constraints
        left_males = sum(1 for p in left if p.gender == 'M')
        left_females = sum(1 for p in left if p.gender == 'F')
        right_males = sum(1 for p in right if p.gender == 'M')
        right_females = sum(1 for p in right if p.gender == 'F')

        # Check gender compliance
        boat_gender = boats[boat_idx].gender
        if boat_gender == 'Open' and (left_females > 0 or right_females > 0):
            print("❌ Gender violation: Open boat has females.")
        elif boat_gender == 'Women' and (left_males > 0 or right_males > 0):
            print("❌ Gender violation: Women's boat has males.")
        elif boat_gender == 'Mixed':
            if left_males + right_males != boats[boat_idx].size // 2 or left_females + right_females != boats[
                boat_idx].size // 2:
                print("❌ Gender violation: Mixed boat doesn't have 50/50 gender split.")

        # Check handedness compliance
        wrong_side = []
        for p in left:
            if p.side == 'R':
                wrong_side.append(f"{p.name} (R on left)")
        for p in right:
            if p.side == 'L':
                wrong_side.append(f"{p.name} (L on right)")

        if wrong_side:
            print(f"⚠️ Handedness violations: {', '.join(wrong_side)}")
        else:
            print("✅ Handedness constraints satisfied")

        print("-" * 30)


# Example usage:
if __name__ == "__main__":
    # Note: 'backend/roster.csv' must exist and have the correct format for this to run.
    data = parse_csv("backend/roster.csv")

    if not data:
        print("Cannot run boat assignment without valid data.")
    else:
        # Example boats
        boats = [
            Boat(size=20, gender="Mixed"),  # 20 people, 10 male, 10 female
            Boat(size=16, gender="Open"),  # 16 people, all male
            Boat(size=12, gender="Women"),  # 12 people, all female
        ]

        try:
            boat_assignments = build_boat(data, boats)

            # Print first half assignments
            print_half("First Half", boat_assignments['first_half'], boats, data)

            # Print second half assignments
            print_half("Second Half", boat_assignments['second_half'], boats, data)

        except (ValueError, IndexError) as e:
            print(f"An error occurred: {e}")
