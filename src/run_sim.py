import os
import time
import mujoco
import mujoco.viewer

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(base_dir, 'config', 'simple_pendulum.xml')
    
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Could not find physics configuration at: {xml_path}")

    print("Compiling static simulation blueprint (mjModel)...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    print("Allocating frame-by-frame state memory (mjData)...")
    data = mujoco.MjData(model)

    print("Launching interactive 3D physics window...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_start_time = data.time
        real_start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()

            # Apply a programmatic torque pulse every 2 seconds
            if int(data.time) % 2 == 0 and (data.time - int(data.time)) < 0.1:
                data.ctrl[0] = 5.0
            else:
                data.ctrl[0] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed_real = time.time() - real_start_time
            elapsed_sim = data.time - sim_start_time
            
            if elapsed_sim > elapsed_real:
                time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
