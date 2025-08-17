import React, { useState, useCallback, useEffect } from 'react';
import { Plus, Trash2, Users, Scale, CheckCircle, AlertCircle, Ship, UserCheck, User } from 'lucide-react';

// People will be loaded from /public/roster.csv at runtime

// Mock optimization function - replace with actual linear programming implementation
const optimizeBoats = (people, boats) => {
  // This is a simplified mock - in production, this would call your Python LP solver
  const assignments = {};
  let usedPeople = new Set();

  boats.forEach((boat, boatIdx) => {
    const availablePeople = people.filter((p, idx) => !usedPeople.has(idx));
    let selectedPeople = [];

    // Simple selection based on constraints
    if (boat.gender === 'Open') {
      selectedPeople = availablePeople.filter(p => p.gender === 'M').slice(0, boat.size);
    } else if (boat.gender === 'Women') {
      selectedPeople = availablePeople.filter(p => p.gender === 'F').slice(0, boat.size);
    } else { // Mixed
      const males = availablePeople.filter(p => p.gender === 'M').slice(0, boat.size / 2);
      const females = availablePeople.filter(p => p.gender === 'F').slice(0, boat.size / 2);
      selectedPeople = [...males, ...females];
    }

    // Mark as used
    selectedPeople.forEach(person => {
      const idx = people.findIndex(p => p.name === person.name);
      usedPeople.add(idx);
    });

    // Split into left and right based on handedness
    const left = selectedPeople.filter(p => p.side === 'L' || (p.side === 'A' && Math.random() < 0.5));
    const right = selectedPeople.filter(p => !left.includes(p));

    // Balance if needed
    while (left.length < boat.size / 2 && right.length > boat.size / 2) {
      const person = right.find(p => p.side === 'A');
      if (person) {
        right.splice(right.indexOf(person), 1);
        left.push(person);
      } else break;
    }
    while (right.length < boat.size / 2 && left.length > boat.size / 2) {
      const person = left.find(p => p.side === 'A');
      if (person) {
        left.splice(left.indexOf(person), 1);
        right.push(person);
      } else break;
    }

    assignments[boatIdx] = { left, right };
  });

  return assignments;
};

const BoatAssignmentManager = () => {
  const [boats, setBoats] = useState([]);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentBoat, setCurrentBoat] = useState({ size: 20, gender: 'Mixed' });
  const [people, setPeople] = useState([]);

  useEffect(() => {
    const loadRoster = async () => {
      try {
        const response = await fetch((process.env.PUBLIC_URL || '') + '/roster.csv');
        const text = await response.text();
        const lines = text.trim().split(/\r?\n/);
        // Remove header
        if (lines.length > 0) lines.shift();
        const parsed = lines
          .map(line => line.trim())
          .filter(Boolean)
          .map(line => {
            const [name, gender, weight, side] = line.split(',');
            return {
              name: name ? name.trim() : '',
              gender: gender ? gender.trim() : '',
              weight: weight ? parseFloat(weight) : 0,
              side: side ? side.trim() : ''
            };
          });
        setPeople(parsed);
      } catch (error) {
        console.error('Failed to load roster.csv', error);
        setPeople([]);
      }
    };
    loadRoster();
  }, []);

  const addBoat = useCallback(() => {
    if (currentBoat.size > 0 && currentBoat.size % 2 === 0) {
      setBoats(prev => [...prev, { ...currentBoat, id: Date.now() }]);
      setCurrentBoat({ size: 20, gender: 'Mixed' });
    }
  }, [currentBoat]);

  const removeBoat = useCallback((id) => {
    setBoats(prev => prev.filter(boat => boat.id !== id));
  }, []);

  const generateAssignments = useCallback(async () => {
    if (boats.length === 0 || people.length === 0) return;

    setIsLoading(true);
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500));

    const assignments = optimizeBoats(people, boats);
    setResults(assignments);
    setIsLoading(false);
  }, [boats, people]);

  const getTotalPeople = () => boats.reduce((sum, boat) => sum + boat.size, 0);
  const getGenderIcon = (gender) => {
    switch (gender) {
      case 'Open': return <User className="w-4 h-4" />;
      case 'Women': return <UserCheck className="w-4 h-4" />;
      case 'Mixed': return <Users className="w-4 h-4" />;
      default: return null;
    }
  };

  const validateBoats = () => {
    const totalPeople = getTotalPeople();
    const availablePeople = people.length;
    return totalPeople <= availablePeople;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex justify-center items-center gap-3 mb-4">
            <Ship className="w-10 h-10 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900">Boat Assignment Manager</h1>
          </div>
          <p className="text-lg text-gray-600">Configure boats and optimize crew assignments</p>
        </div>

        {!results ? (
          /* Configuration Phase */
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Boat Builder */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
                <Plus className="w-6 h-6 text-blue-600" />
                Add Boat
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Boat Size (must be even)
                  </label>
                  <input
                    type="number"
                    value={currentBoat.size}
                    onChange={(e) => setCurrentBoat(prev => ({ ...prev, size: parseInt(e.target.value) || 0 }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="2"
                    step="2"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Gender Requirement
                  </label>
                  <select
                    value={currentBoat.gender}
                    onChange={(e) => setCurrentBoat(prev => ({ ...prev, gender: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="Mixed">Mixed (Half Male, Half Female)</option>
                    <option value="Open">Open (Males Only)</option>
                    <option value="Women">Women (Females Only)</option>
                  </select>
                </div>

                <button
                  onClick={addBoat}
                  disabled={currentBoat.size <= 0 || currentBoat.size % 2 !== 0}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  Add Boat
                </button>
              </div>
            </div>

            {/* Boat List */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
                <Ship className="w-6 h-6 text-blue-600" />
                Configured Boats ({boats.length})
              </h2>

              <div className="space-y-3 mb-6 max-h-64 overflow-y-auto">
                {boats.map((boat, index) => (
                  <div key={boat.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center gap-3">
                      {getGenderIcon(boat.gender)}
                      <div>
                        <div className="font-medium text-gray-900">
                          Boat {index + 1}
                        </div>
                        <div className="text-sm text-gray-600">
                          {boat.size} people • {boat.gender}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => removeBoat(boat.id)}
                      className="text-red-600 hover:text-red-800 transition-colors"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                ))}
                {boats.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No boats configured yet. Add a boat to get started.
                  </div>
                )}
              </div>

              {boats.length > 0 && (
                <div className="border-t pt-4">
                  <div className="flex justify-between items-center mb-4">
                    <div className="text-sm text-gray-600">
                      Total People Required: <span className="font-semibold">{getTotalPeople()}</span> / {people.length} available
                    </div>
                    {!validateBoats() && (
                      <div className="flex items-center gap-1 text-red-600">
                        <AlertCircle className="w-4 h-4" />
                        <span className="text-sm">Not enough people</span>
                      </div>
                    )}
                  </div>

                  <button
                    onClick={generateAssignments}
                    disabled={!validateBoats() || isLoading}
                    className="w-full bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
                  >
                    {isLoading ? 'Optimizing Assignments...' : 'Generate Optimal Assignments'}
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Results Phase */
          <div className="space-y-6">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <CheckCircle className="w-8 h-8 text-green-600" />
                Assignment Results
              </h2>
              <button
                onClick={() => setResults(null)}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Configure New Boats
              </button>
            </div>

            <div className="grid gap-6">
              {boats.map((boat, boatIdx) => {
                const assignment = results[boatIdx];
                if (!assignment) return null;

                const leftWeight = assignment.left.reduce((sum, p) => sum + p.weight, 0);
                const rightWeight = assignment.right.reduce((sum, p) => sum + p.weight, 0);
                const weightDiff = Math.abs(leftWeight - rightWeight);

                return (
                  <div key={boat.id} className="bg-white rounded-xl shadow-lg overflow-hidden">
                    <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <Ship className="w-8 h-8" />
                          <div>
                            <h3 className="text-2xl font-bold">Boat {boatIdx + 1}</h3>
                            <p className="text-blue-100">{boat.gender} • {boat.size} people</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="flex items-center gap-2 text-blue-100">
                            <Scale className="w-5 h-5" />
                            <span>Weight Difference</span>
                          </div>
                          <div className="text-2xl font-bold">{weightDiff.toFixed(1)} lbs</div>
                        </div>
                      </div>
                    </div>

                    <div className="p-6">
                      <div className="grid md:grid-cols-2 gap-6">
                        {/* Left Side */}
                        <div className="space-y-3">
                          <h4 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                            Left Side ({assignment.left.length})
                          </h4>
                          <div className="text-sm text-gray-600 mb-3">
                            Total Weight: {leftWeight.toFixed(1)} lbs
                          </div>
                          <div className="space-y-2 max-h-64 overflow-y-auto">
                            {assignment.left.map((person, idx) => (
                              <div key={idx} className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                                <div>
                                  <div className="font-medium text-gray-900">{person.name}</div>
                                  <div className="text-sm text-gray-600">
                                    {person.gender} • {person.side === 'A' ? 'Ambidextrous' : person.side === 'L' ? 'Left-handed' : 'Right-handed'}
                                  </div>
                                </div>
                                <div className="text-sm font-medium text-gray-900">
                                  {person.weight} lbs
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Right Side */}
                        <div className="space-y-3">
                          <h4 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                            Right Side ({assignment.right.length})
                          </h4>
                          <div className="text-sm text-gray-600 mb-3">
                            Total Weight: {rightWeight.toFixed(1)} lbs
                          </div>
                          <div className="space-y-2 max-h-64 overflow-y-auto">
                            {assignment.right.map((person, idx) => (
                              <div key={idx} className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                                <div>
                                  <div className="font-medium text-gray-900">{person.name}</div>
                                  <div className="text-sm text-gray-600">
                                    {person.gender} • {person.side === 'A' ? 'Ambidextrous' : person.side === 'L' ? 'Left-handed' : 'Right-handed'}
                                  </div>
                                </div>
                                <div className="text-sm font-medium text-gray-900">
                                  {person.weight} lbs
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Statistics */}
                      <div className="mt-6 pt-6 border-t border-gray-200">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center">
                            <div className="text-2xl font-bold text-blue-600">
                              {assignment.left.filter(p => p.gender === 'M').length + assignment.right.filter(p => p.gender === 'M').length}
                            </div>
                            <div className="text-sm text-gray-600">Males</div>
                          </div>
                          <div className="text-center">
                            <div className="text-2xl font-bold text-pink-600">
                              {assignment.left.filter(p => p.gender === 'F').length + assignment.right.filter(p => p.gender === 'F').length}
                            </div>
                            <div className="text-sm text-gray-600">Females</div>
                          </div>
                          <div className="text-center">
                            <div className="text-2xl font-bold text-purple-600">
                              {(leftWeight + rightWeight).toFixed(1)}
                            </div>
                            <div className="text-sm text-gray-600">Total Weight</div>
                          </div>
                          <div className="text-center">
                            <div className={`text-2xl font-bold ${weightDiff < 10 ? 'text-green-600' : weightDiff < 25 ? 'text-yellow-600' : 'text-red-600'}`}>
                              {((weightDiff / (leftWeight + rightWeight)) * 100).toFixed(1)}%
                            </div>
                            <div className="text-sm text-gray-600">Imbalance</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BoatAssignmentManager;