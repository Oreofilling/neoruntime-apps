from hailo_ipc_sdk import InferenceClient, EventClient
import time

def main():
    # Initialize clients.
    inf = InferenceClient()
    events = EventClient()

    print("Starting person detection...")

    # Subscribe to video inference results.
    for frame_seq, result in inf.subscribe(
        stream="main",
        model="person_vehicle_v1",
        fps=10
    ):
        # Count detected people.
        person_count = len([
            obj for obj in result.objects
            if obj.label == "person"
        ])

        if person_count > 0:
            print(f"Frame {frame_seq}: detected {person_count} person(s)")

            # Publish an alert event.
            events.publish("app/alert", {
                "type": "person_detected",
                "count": person_count,
                "timestamp": time.time()
            })

if __name__ == "__main__":
    main()
