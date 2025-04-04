### **BETA TEST PLAN – Neptune AI Drowning Recognition**

## **1. Core Functionalities for Beta Version**
For this Beta Test Plan, we focus on Neptune’s essential features to ensure stability and reliability in real-world beach patrol conditions.

| **Feature Name**  | **Description** | **Priority (High/Medium/Low)** | **Changes Since Tech3** |
|-------------------|---------------|--------------------------------|--------------------------|
| Neptune Water Surface Detection (NWSD) | Identifies and tracks the water’s surface boundary under varying sea and lighting conditions. | Low | New feature |
| Neptune Human Detection (NHD) | Detects and locates people on the beach and in the water with high accuracy. | Medium | Division of the work between NHD and NDD |
| Neptune Drowning Detection (NDD) | Recognizes potential drowning events in real time and triggers alerts. | High | N/A |
| Live Detection Display | Streams real-time outputs from NWSD, NHD, and NDD with overlays. | Low | N/A |
| Audible Alerts | Issues sound notifications (e.g., "Drowning Alert!") for emergencies. | Low | N/A |
| 2D Homography Beach Map | Projects detections onto a top-down beach map with color-coded markers. | High | New feature |

---

## 2. Definition of Beta Testing Scenarios
### **2.1 User Roles**

| **Role Name**  | **Description** |
|---------------|---------------|
| Beach Lifeguard       | Monitors the beach, responds to alerts, and performs rescues. |
| System Admin       | Manages the Neptune system and hardware. |

### **2.2 Test Scenarios**

#### **Scenario 1: Water Surface Detection Accuracy**
- **Role Involved:** System Admin
- **Objective:** Test the Neptune Water Surface Detection (NWSD) model's ability to correctly segment water areas.
- **Preconditions:** System camera is positioned correctly.
- **Test Steps:**
  1. Capture video feed in different lighting and weather conditions.
  2. Run NWSD on live footage.
  3. Compare detected water areas with ground truth images.
- **Expected Outcome:** The system correctly identifies water surfaces with at least 90% accuracy.

#### **Scenario 2: Human Detection in the Water**
- **Role Involved:** Beach Lifeguard
- **Objective:** Evaluate the Neptune Human Detection (NHD) model in identifying people in the sea.
- **Preconditions:** At least one person is in the water, at varying distances from the camera.
- **Test Steps:**
  1. Monitor video feed with people swimming at different locations.
  2. Run NHD and log detected human figures.
  3. Compare results with actual presence in the water.
- **Expected Outcome:** The system correctly detects humans in 95% of cases without false positives.

#### **Scenario 3: Drowning Pattern Detection & Alert System**
- **Role Involved:** Beach Lifeguard
- **Objective:** Validate the Neptune Drowning Detection (NDD) model's real-time alert system.
- **Preconditions:** A controlled test with a swimmer mimicking drowning behavior.
- **Test Steps:**
  1. Simulate a drowning event in view of the camera.
  2. Observe if the system generates a "Drowning Alert!" notification with sound.
  3. Verify that the lifeguard receives the alert in under 5 seconds.
- **Expected Outcome:** A timely and accurate alert is issued when drowning behavior is detected.

#### **Scenario 4: Hardware Performance Under Environmental Stress**
- **Role Involved:** System Admin
- **Objective:** Test Neptune’s hardware stability under extreme weather conditions.
- **Preconditions:** System is set up on the beach and exposed to elements.
- **Test Steps:**
  1. Operate the system in direct sunlight, high humidity.
  2. Measure processing speed and temperature stability.
  3. Log any overheating, lag, or hardware failures.
- **Expected Outcome:** The system runs continuously without crashes or major delays.


___

## 3. Coverage of Key User Journeys

### Journey: Normal Surveillance
**Objective:** Ensure effective monitoring without the need to constantly look at the screen.

**Key Steps:**
- Turning on the Neptune system upon arrival at the post.
- Initial verification that the system is functioning (camera connection, active detection).
- Passive monitoring: the lifeguard does not need to look at the screen unless an alert is triggered.
- In case of an audible alert:
  - The lifeguard listens to the type of alert issued (e.g., "Drowning Alert!").
  - They can choose to check the video feed or act immediately based on the situation.
  - If the danger is not immediately visible (e.g., a crowded beach), the lifeguard may use the video feed to precisely locate the affected area.
- Logging out and shutting down the software at the end of service.

**Expected Outcome:**
- Smooth monitoring without unnecessary distractions.
- Audible alerts are sufficient to warn of problems.
- The video feed serves as a support tool in cases where the location of the danger is uncertain, particularly on crowded beaches.

**Possible Failure Points:**
- The audible alert does not trigger despite a detected danger.
- An alert is triggered but is not loud enough to be heard.
- Too many false alerts disrupt monitoring and require excessive screen-checking.
- On a crowded beach, the video feed must provide sufficient clarity to allow quick identification of the danger.

---

### Journey: Responding to a Drowning Alert
**Objective:** Respond quickly to a drowning detected by Neptune.

**Key Steps:**
- Passive monitoring by the lifeguard (not constantly watching the screen).
- Neptune triggers an audible alert: "Drowning Alert!".
- Listening and identifying the type of alert.
- If the danger is immediately visible, direct intervention.
- If the danger is unclear, a quick check of the video feed to locate the distressed individual.
- Immediate movement towards the identified area and rescue.
- Deactivation of the alert once the intervention is complete.

**Expected Outcome:**
- The alert enables an immediate and effective reaction.
- The video feed is used only when necessary to help locate the danger.
- The intervention is quick and well-coordinated.

**Possible Failure Points:**
- The alert does not trigger despite an ongoing drowning.
- The alert is triggered but is inaudible due to ambient noise.
- The video feed is blurry or delayed, making localization difficult.

---

### Journey: Managing a False Alert
**Objective:** Identify and handle false alerts to avoid unnecessary interventions.

**Key Steps:**
- Passive monitoring by the lifeguard.
- Neptune triggers an audible alert.
- Observing the behavior of the concerned individual:
  - Checking the video feed if in doubt.
  - Direct visual verification on the beach.
- If it is a false alert, manual validation in Neptune.
- Adjusting parameters to prevent similar errors in the future.

**Expected Outcome:**
- Reducing unnecessary interruptions caused by false positives.
- Continuous improvement of the detection model.
- Increased system efficiency without unnecessary alerts for minor events.

**Possible Failure Points:**
- Too many false alerts fatigue the lifeguard and reduce their responsiveness.
- The Neptune algorithm does not learn enough from mistakes.
- A false alert is misidentified, and a dangerous situation is ignored.

---

### Journey: Surveillance During Exceptional Crowds
**Objective:** Ensure effective detection when the beach is overcrowded.

**Key Steps:**
- Turning on Neptune and verifying the cameras.
- Passive monitoring, with the lifeguard not constantly watching the screen.
- In case of an audible alert, quick danger analysis:
  - Direct visual verification to locate the endangered person.
  - If the crowd prevents immediate identification, checking the video feed to find the individual in question.
- Immediate intervention after precise localization.

**Expected Outcome:**
- Neptune operates effectively, even when the beach is heavily crowded.
- The video feed assists in locating a person in distress when hidden within the crowd.
- Alerts remain precise, avoiding unnecessary overload due to excessive human movement.

**Possible Failure Points:**
- Too much human movement confuses the detection, increasing false positives.
- The video feed becomes harder to analyze when the beach is densely packed.
- A real danger is lost within the crowd and detected too late.

___

## **4. Success Criteria**

| Feature           | Criterion                | Formula / Method                                | Unit | Minimum Threshold | Measurement Method                                  | Frequency           |
|-------------------|--------------------------|-------------------------------------------------|------|-------------------|-----------------------------------------------------|---------------------|
| NWSD              | Detection Accuracy       | TP / (TP + FP)                                  | %    | ≥ 80%             | Manual log analysis                                 | After each update   |
| NWSD              | Latency                  | t_alert – t_frame                               | ms   | ≤ 500             | Software timestamp                                  | Continuous          |
| NHD               | Recall                   | TP / (TP + FN)                                  | %    | ≥ 80%             | Manual comparison vs model output                   | 10 scenarios        |
| NHD               | False Positive Rate      | FP / (FP + TN)                                  | %    | ≤ 3%              | Event logs                                         | Continuous          |
| NDD               | Detection Time           | t_alert – t_incident                            | s    | ≤ 3               | Automated simulation script                        | 5 tests/week        |
| NDD               | False Alarm Rate         | Number of false alarms ÷ total alerts           | %    | ≤ 5%              | Manual review                                      | After each test     |
| Live Display      | Refresh Rate             | 1 / (update interval)                           | Hz   | ≥ 1               | Profiling tool                                     | Continuous          |
| Sound Alerts      | Alert Volume             | Measured SPL level                              | dB   | ≥ 85              | Sound level meter                                  | One‑time test       |
| 2D Map            | Homography Accuracy      | Mean reprojection error                         | m    | ≤ 1               | Average distance between projected and true points | 10 test points      |
| System Usability  | SUS Score                | Standard System Usability Scale                 | score| ≥ 80              | SUS questionnaire filled by sea guards             | After beta test     |
| Alert Acknowledgement | Task Completion Time | t_ack – t_alert                                 | s    | ≤ 5               | Software timestamp                                 | Continuous          |

## **5. Known Issues & Limitations**
| **Issue** | **Description** | **Impact** | **Planned Fix? (Yes/No)** |
|----------|---------------|----------|----------------|
| False Drowning Alerts | Some activities (e.g., diving) are misclassified as drowning. | Medium | Yes |
| No Drowning Alerts | No alert in case of drowning. | High | Yes |
| Human Detection from a far distance | Detecting a human from a far distance is very challenging (can be confuse with other objects or not detected) | High | Yes |

---

## **5. Conclusion**
This Beta Test Plan ensures that Neptune AI’s drowning detection and monitoring capabilities are reliable in real-world conditions. The outlined test scenarios, success criteria, and mitigation plans aim to optimize performance and user experience, making Neptune an invaluable tool for beach safety.
