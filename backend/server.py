from flask import Flask, request, jsonify
from flask_cors import CORS
from main import parse_csv, Boat, build_boat
import csv
import os
from shutil import copyfile

app = Flask(__name__)
CORS(app)

# File paths
ROSTER_FILE = 'roster.csv'
ACTIVE_ROSTER_FILE = 'roster-active.csv'


# Initialize roster-active.csv if it doesn't exist
def initialize_active_roster():
    if not os.path.exists(ACTIVE_ROSTER_FILE):
        try:
            copyfile(ROSTER_FILE, ACTIVE_ROSTER_FILE)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {ROSTER_FILE} not found. Please ensure it exists.")


@app.route('/people', methods=['GET'])
def list_people():
    try:
        initialize_active_roster()
        all_people = parse_csv(ROSTER_FILE)
        active_people = parse_csv(ACTIVE_ROSTER_FILE)
        active_names = {p.name for p in active_people}

        serialized = [
            {
                'name': p.name,
                'gender': p.gender,
                'weight': p.weight,
                'side': p.side,
                'active': p.name in active_names
            }
            for p in all_people
        ]
        return jsonify({'people': serialized, 'count': len(serialized)}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/people/<name>/toggle', methods=['POST'])
def toggle_person(name):
    try:
        initialize_active_roster()
        all_people = parse_csv(ROSTER_FILE)
        active_people = parse_csv(ACTIVE_ROSTER_FILE)
        active_names = {p.name for p in active_people}

        if name not in {p.name for p in all_people}:
            return jsonify({'error': f"Person '{name}' not found in roster"}), 404

        if name in active_names:
            # Remove person from active roster
            active_people = [p for p in active_people if p.name != name]
        else:
            # Add person to active roster
            person = next(p for p in all_people if p.name == name)
            active_people.append(person)

        # Write updated active roster to file
        with open(ACTIVE_ROSTER_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Name', 'Gender', 'Side', 'Weight'])
            writer.writeheader()
            for p in active_people:
                writer.writerow({
                    'Name': p.name,
                    'Gender': p.gender,
                    'Side': p.side,
                    'Weight': p.weight
                })

        return jsonify({'message': f"Person '{name}' toggled successfully", 'active': name not in active_names}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/people/toggle-all', methods=['POST'])
def toggle_all_people():
    try:
        initialize_active_roster()
        payload = request.get_json(force=True) or {}
        set_active = payload.get('active', True)

        if set_active:
            # Copy all people from roster.csv to roster-active.csv
            copyfile(ROSTER_FILE, ACTIVE_ROSTER_FILE)
            message = "All people activated"
        else:
            # Clear roster-active.csv (write only header)
            with open(ACTIVE_ROSTER_FILE, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['Name', 'Gender', 'Side', 'Weight'])
                writer.writeheader()
            message = "All people deactivated"

        return jsonify({'message': message}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/assignments', methods=['POST'])
def create_assignments():
    try:
        initialize_active_roster()
        payload = request.get_json(force=True) or {}
        boats_payload = payload.get('boats', [])

        boats = []
        for boat in boats_payload:
            size = int(boat.get('size', 0))
            gender = boat.get('gender', 'Mixed')
            boats.append(Boat(size=size, gender=gender))

        people = parse_csv(ACTIVE_ROSTER_FILE)  # Use active roster for assignments

        result = build_boat(people, boats)

        # Serialize dataclasses to dicts for JSON response
        serialized = {
            'first_half': {},
            'second_half': {}
        }

        # Serialize first_half
        for idx in result['first_half']:
            left_side, right_side = result['first_half'][idx]
            serialized['first_half'][str(idx)] = {
                'left': [
                    {
                        'name': p.name,
                        'gender': p.gender,
                        'weight': p.weight,
                        'side': p.side,
                    }
                    for p in left_side
                ],
                'right': [
                    {
                        'name': p.name,
                        'gender': p.gender,
                        'weight': p.weight,
                        'side': p.side,
                    }
                    for p in right_side
                ],
            }

        # Serialize second_half
        for idx in result['second_half']:
            left_side, right_side = result['second_half'][idx]
            serialized['second_half'][str(idx)] = {
                'left': [
                    {
                        'name': p.name,
                        'gender': p.gender,
                        'weight': p.weight,
                        'side': p.side,
                    }
                    for p in left_side
                ],
                'right': [
                    {
                        'name': p.name,
                        'gender': p.gender,
                        'weight': p.weight,
                        'side': p.side,
                    }
                    for p in right_side
                ],
            }

        return jsonify({'assignments': serialized}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)