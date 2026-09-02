1. Requirements and mission definition
- Translate an informal objective into quantitative requirements and constraints.
- Define a representative mission profile: takeoff, climb, cruise, loiter, descent, landing, reserve.
- Distinguish requirements, design variables, constraints, and objectives.
- Identify which requirement sizes each subsystem.
- Recognize conflicting objectives: range, endurance, speed, payload, cost, stability, and structural weight.
- Select appropriate design margins without arbitrarily oversizing the aircraft.
- Understand the iterative nature of aircraft sizing.
2. Atmosphere and operating altitude
- Use a standard atmosphere model to determine density, pressure, temperature, and speed of sound.
- Distinguish indicated, equivalent, and true airspeed.
- Explain altitude tradeoffs involving:
  - Dynamic pressure and required true airspeed
  - Reynolds number
  - Parasite and induced drag
  - Propeller performance
  - Motor cooling
  - Stall speed
  - Available and required power
- Choose a cruise altitude based on mission and vehicle characteristics rather than assuming “higher is better.”
- Understand wind effects on ground speed, range, and mission feasibility.
3. Aircraft-level sizing
- Construct a mass/weight budget and track uncertainty and growth.
- Estimate component masses using historical or physics-based models.
- Understand the coupling among weight, wing area, propulsion, battery/fuel, and structure.
- Select wing loading and power loading or thrust loading.
- Use constraint diagrams to combine:
  - Stall speed
  - Takeoff distance
  - Climb rate or climb gradient
  - Cruise speed
  - Ceiling
  - Turning performance
- Identify feasible design space and select a defensible design point.
- Understand why sizing is an iterative convergence problem.
4. Aerodynamic fundamentals
- Apply lift, drag, and moment coefficients consistently.
- Distinguish dimensional forces from nondimensional coefficients.
- Understand Reynolds-number and Mach-number effects.
- Estimate typical values of:
  - Lift-curve slope
  - Maximum lift coefficient
  - Zero-lift drag coefficient
  - Oswald efficiency factor
  - Lift-to-drag ratio
- Recognize the limitations of handbook correlations, CFD, wind-tunnel data, and low-fidelity models.
Drag and aerodynamic efficiency
- Build a drag polar:\[
  C_D=C_{D0}+kC_L^2
  \]
- Distinguish skin-friction, form, interference, induced, and wave drag.
- Explain the balance between parasite and induced drag.
- Determine the conditions for:
  - Maximum \(L/D\)
  - Minimum drag
  - Minimum power required
- Understand that maximum range and maximum endurance generally occur at different operating conditions.
- Estimate wetted area and component drag buildup.
- Account for trim drag and cooling drag.
5. Airfoil selection
- Interpret airfoil polars rather than selecting an airfoil from its name or maximum \(L/D\).
- Understand:
  - Lift-curve slope and zero-lift angle
  - Maximum lift and stall behavior
  - Drag bucket and transition sensitivity
  - Reynolds-number sensitivity
  - Surface-roughness sensitivity
  - Pitching moment
  - Thickness and structural depth
- Recognize that airfoil \(L/D\) is not aircraft \(L/D\).
- Select an operating \(C_L\) range consistent with the mission.
- Balance low drag against stall behavior, pitching moment, manufacturability, and structural needs.
- Know when laminar-flow predictions are unlikely to be realized in practice.
6. Wing geometry and loading
- Select wing area, span, aspect ratio, sweep, taper, twist, and dihedral.
- Understand span efficiency and induced drag.
- Explain why an exactly elliptical planform is not necessarily the best wing.
- Design a near-elliptical span loading using taper and twist.
- Understand geometric versus aerodynamic twist.
- Avoid premature tip stall and preserve aileron effectiveness.
- Evaluate root bending moment as well as aerodynamic performance.
- Understand the aerodynamic–structural tradeoff in aspect ratio.
- Estimate the effects of:
  - Fuselage interference
  - Wingtip geometry
  - Control-surface cutouts
  - Propeller slipstream
  - Ground effect
- Choose high-lift devices when appropriate and estimate their effect on \(C_{L_{\max}}\), drag, and pitching moment.
7. Trim, stability, and control
A student should distinguish clearly among equilibrium, trim, static stability, dynamic stability, and controllability.
Longitudinal
- Locate the aerodynamic center and neutral point.
- Calculate static margin.
- Understand the effects of CG location.
- Size the horizontal tail using tail volume coefficient and moment balance.
- Account for downwash and tail dynamic-pressure ratio.
- Select tail incidence and wing incidence.
- Determine elevator authority for:
  - Trim
  - Rotation
  - Flare
  - Low-speed operation
- Understand the tradeoff between stability, control authority, and trim drag.
- Identify short-period and phugoid behavior.
Lateral-directional
- Understand directional stability, weathercock stability, and vertical-tail sizing.
- Understand dihedral effect and contributions from wing position, sweep, and dihedral.
- Size rudder and ailerons for anticipated maneuvers and disturbances.
- Recognize roll, spiral, and Dutch-roll modes.
- Evaluate control authority in crosswind, asymmetric thrust, or other critical conditions.
- Understand adverse yaw and possible mitigation.
Handling and controllability
- Construct allowable CG limits from stability and control requirements.
- Recognize control saturation and rate limits.
- Relate stability derivatives to physical aircraft features.
- Understand when active control can supplement—but not magically eliminate—poor vehicle design.
8. Propulsion and energy systems
General matching
- Distinguish thrust available from thrust required and power available from power required.
- Match the propulsion system to takeoff, climb, cruise, and endurance requirements.
- Understand installed performance rather than relying on component ratings.
- Account for efficiency variation across the mission.
- Understand thermal limits and cooling requirements.
Electric aircraft
- Understand motor \(K_v\), torque constant, current, voltage, resistance, and efficiency.
- Interpret motor maps and operating limits.
- Match battery voltage, motor, controller, and propeller.
- Understand battery:
  - Cell count
  - Capacity
  - C-rating
  - Internal resistance
  - Voltage sag
  - Usable energy
  - State of charge
  - Cycle life and safety
- Recognize the difference between peak and continuous ratings.
- Estimate electrical and thermal losses.
- Understand why adding battery can eventually reduce rather than increase useful endurance.
Propellers
- Understand advance ratio, tip speed, pitch, diameter, solidity, and blade count.
- Use propeller coefficients \(C_T\), \(C_P\), and efficiency.
- Distinguish static thrust from cruise performance.
- Match propeller load to motor torque and speed.
- Understand diameter–RPM–noise–clearance tradeoffs.
- Account for off-design performance and propeller–airframe interactions.
- Recognize when momentum theory, blade-element methods, or empirical data are appropriate.
9. Flight performance
Students should be able to derive performance from the drag polar and propulsion model, not merely enter numbers into separate formulas.
- Stall speed and effects of weight, altitude, bank angle, and \(C_{L_{\max}}\)
- Maximum and minimum level-flight speed
- Power-required and thrust-required curves
- Maximum range and endurance
- Best-glide speed and glide ratio
- Sink rate and minimum-sink speed
- Rate and angle of climb
- Service and absolute ceiling
- Takeoff and landing distance
- Accelerated flight and turning performance
- Load factor, turn radius, and turn rate
- Effects of weight, altitude, wind, and configuration
- Energy methods for understanding climb, acceleration, and maneuvering
- Mission simulation with changing mass, battery voltage, or atmospheric conditions
A particularly important concept is that “best speed” depends on the objective: best glide, minimum sink, maximum range, maximum endurance, maximum climb angle, and maximum climb rate are not generally the same speed.
10. Loads and structures
- Construct maneuver and gust \(V\)-\(n\) diagrams.
- Understand limit load, ultimate load, and factor of safety.
- Identify critical load cases rather than designing only for nominal cruise.
- Determine aerodynamic span loads and resulting shear, bending moment, and torsion.
- Understand load paths through wings, spars, skins, joints, fuselage, and landing gear.
- Estimate bending, shear, torsional, and bearing stresses.
- Check:
  - Yield and ultimate strength
  - Buckling
  - Deflection
  - Fatigue where relevant
  - Joint and fastener failure
- Understand stiffness requirements in addition to strength requirements.
- Recognize structural–aerodynamic coupling, including twist and aeroelastic effects.
- Balance structural mass against span, aspect ratio, thickness, and load factor.
- Design for manufacturing tolerances, damage, and repairability.
11. Configuration and integration
- Choose among conventional tail, T-tail, V-tail, canard, flying wing, and other configurations.
- Position the wing, payload, battery/fuel, landing gear, and propulsion system.
- Manage CG travel throughout the mission and across payload configurations.
- Ensure geometric compatibility:
  - Propeller clearance
  - Control-surface motion
  - Ground clearance
  - Rotation angle
  - Tail-strike margin
  - Internal packaging
- Understand landing-gear placement, tip-back angle, overturn angle, and ground handling.
- Account for wiring, actuators, avionics, cooling, access, assembly, and maintenance.
- Recognize integration effects that isolated subsystem analyses miss.
12. Modeling, verification, and uncertainty
This deserves explicit coverage because it separates credible design from spreadsheet numerology.
- Use dimensional analysis and check units.
- Perform order-of-magnitude and limiting-case checks.
- Distinguish model inputs, assumptions, calibration parameters, and outputs.
- Select an analysis fidelity appropriate to the decision.
- Validate low-fidelity predictions against experimental or published data.
- Quantify sensitivity to uncertain inputs.
- Propagate uncertainty or at least provide realistic performance bounds.
- Avoid false precision.
- Identify which assumptions drive the design conclusion.
- Plan tests that reduce the most consequential uncertainties.
- Compare predictions with flight-test data and update the model.
13. Design decision-making
- Perform parameter sweeps rather than analyzing only one design.
- Understand local versus system-level optimization.
- Use trade studies with meaningful metrics.
- Avoid double-counting advantages or penalties.
- Recognize when a constraint is active.
- Explain why a chosen design is preferable to nearby alternatives.
- Document assumptions and maintain traceability from requirements to design decisions.
- Communicate results through clear plots, tables, drawings, and concise engineering arguments.
Suggested homework progression
A coherent homework sequence could build one aircraft throughout the semester:
1. Define mission, requirements, and design metrics.
2. Build atmosphere and mission-analysis tools.
3. Choose wing loading and power loading using constraint analysis.
4. Select an airfoil from Reynolds-number-appropriate data.
5. Design the wing planform, twist, and span loading.
6. Build an aircraft drag polar and estimate \(L/D\).
7. Size and match the motor, propeller, controller, and battery.
8. Predict takeoff, climb, cruise, range, endurance, and glide performance.
9. Locate the neutral point, select CG range, and size the tail and controls.
10. Construct \(V\)-\(n\) diagrams and define critical structural load cases.
11. Size the main wing structure and estimate deflection and mass.
12. Integrate the design, update the weight and CG budget, and repeat the analyses.
13. Perform sensitivity and uncertainty studies.
14. Present a final design review supported by a mission simulation and verified design margins.