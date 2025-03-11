
# Changing contact between bodies with bitmasking

body_contype = 2
body_conaffinity = 1
thigh_contype = 2
thigh_conaffinity = 1
shin_contype = 2
shin_conaffinity = 1
wheel_contype = 4
wheel_conaffinity = 5
world_contype = 1
world_conaffinity = 1

def check_contact(contype1, conaffinity1, contype2, conaffinity2):
    contact = (contype1 & conaffinity2) or (contype2 & conaffinity1)
    return contact

# Check contact between body and thigh
body_thigh = check_contact(body_contype, body_conaffinity, thigh_contype, thigh_conaffinity)
if(body_thigh == True):
    print("Contact between body and thigh")
    print(body_thigh)
else:
    print("No contact between body and thigh")
    print(body_thigh)

# Check contact between body and thigh
thigh_shin = check_contact(thigh_contype, thigh_conaffinity, shin_contype, shin_conaffinity)
if(thigh_shin == True):
    print("Contact between thigh and shin")
    print(thigh_shin)
else:
    print("No contact between thigh and shin")
    print(thigh_shin)

# Check contact between shin and wheel
shin_wheel = check_contact(shin_contype, shin_conaffinity, wheel_contype, wheel_conaffinity)
if(shin_wheel == True):
    print("Contact between shin and wheel")
    print(shin_wheel)
else:    
    print("No contact between shin and wheel")
    print(shin_wheel)

# Check contact between wheel and world
wheel_world = check_contact(wheel_contype, wheel_conaffinity, world_contype, world_conaffinity)
if(wheel_world == True):
    print("Contact between wheel and world")
    print(wheel_world)
else:
    print("No contact between wheel and world")
    print(wheel_world)

# Check contact between shin and world
shin_world = check_contact(shin_contype, shin_conaffinity, world_contype, world_conaffinity)
if(shin_world == True):
    print("Contact between shin and world")
    print(shin_world)
else:
    print("No contact between shin and world")
    print(shin_world)

# Check contact between two wheels:
wheel_wheel = check_contact(wheel_contype, wheel_conaffinity, wheel_contype, wheel_conaffinity)
if(wheel_wheel == True):
    print("Contact between two wheels")
    print(wheel_wheel)
else:    
    print("No contact between two wheels")
    print(wheel_wheel)