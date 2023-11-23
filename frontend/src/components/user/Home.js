import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
} from 'react-native';
import React, {useState} from 'react';
import {Slider, Icon} from '@rneui/themed';
import {useNavigation} from '@react-navigation/native';
import CheckBox from 'react-native-check-box';

const Home = ({route}) => {
  const navigation = useNavigation();
  const {userD} = route.params;

  const [bedrooms, setBedrooms] = useState('No');
  const [propertySize, setPropertySize] = useState('1000');
  const [resale, setResale] = useState('No');
  const [maintenanceStaff, setMaintenanceStaff] = useState('No');
  const [gymnasium, setGymnasium] = useState('No');
  const [swimmingPool, setSwimmingPool] = useState('No');
  const [landscapedGardens, setLandscapedGardens] = useState('No');
  const [joggingTrack, setJoggingTrack] = useState('No');
  const [rainWaterHarvesting, setRainWaterHarvesting] = useState('No');
  const [indoorGames, setIndoorGames] = useState('No');
  const [shoppingmall, setShoppingMall] = useState('No');
  const [interCom, SetInterCom] = useState('No');
  const [sportsFacility, setSportsFacility] = useState('No');
  const [atm, setAtm] = useState('No');
  const [clubHouse, setClubHouse] = useState('No');
  const [school, setSchool] = useState('No');
  const [security, SetSecurity] = useState('No');
  const [powerBackup, setPowerBackup] = useState('No');
  const [carparking, setCarParking] = useState('No');
  const [staffQuarter, setStaffQuarter] = useState('No');
  const [cafteria, setCafteria] = useState('No');
  const [multiPurposeRoom, setMultiPurposeRoom] = useState('No');
  const [hospital, setHospital] = useState('No');
  const [washingMachine,setWashingMachine]=useState('No')
  const [gasConnection, setGasConnection] = useState('No');
  const [AC, setAC] = useState('No');
  const [wifi, setWifi] = useState('No');
  const [childrenArea, setChildrenArea] = useState('No');
  const [lift, setLift] = useState('No');
  const [bed, setBed] = useState('No');
  const [microWaveOwen, setMIcroWaveOwen] = useState('No');
  const [TV, setTV] = useState('No');
  const [sofa, setSofa] = useState('No');
  const [wardrobe, setWardrobe] = useState('No');

  const [date, setDate] = useState(new Date());
  const [isChecked, setIsChecked] = useState(false);

  const color = () => {};

  const handleSubmit = async () => {
    const propertyData = {
      price: 26500,
      Area : propertySize,
      Bedrooms: bedrooms,
      Resale: resale,
      MaintenanceStaff: maintenanceStaff,
      Gymnasium: gymnasium,
      SwimmingPool:swimmingPool,
      LandscapedGardens:landscapedGardens,
      JoggingTrack:joggingTrack,
      RainWaterHarvesting:rainWaterHarvesting,
      IndoorGames:indoorGames,
      ShoppingMall:shoppingmall,
      InterCom:interCom,
      sportsFacility:sportsFacility,
      ATM:atm,
      ClubHouse:clubHouse,
      School:school,
      Security:security,
      powerBackup:powerBackup,
      CarParking:carparking,
      StaffQuarter:staffQuarter,
      Cafteria:cafteria,
      MultiPurposeRoom:multiPurposeRoom,
      Hospital:hospital,
      WashingMachine:washingMachine,
      GasConnection:gasConnection,
      AC:AC,
      Wifi:wifi,
      Childrenplayarea:childrenArea,
      LiftAvailable:lift,
      Bed:bed,
      Microwave:microWaveOwen,
      TV:TV,
      Sofa:sofa,
      Wardrobe:wardrobe,
      // location:userD.place,
      userName: userD.name,
      phone: userD.phone,
      userId: userD.email,
      status: 'Available',
      Date: date.toISOString(),
    };
    console.log(propertyData);
    navigation.navigate('Property', {userD, propertyData});
  };
  height: 40;
  return (
    <View style={styles.container}>
      <View style={{backgroundColor: 'white', marginBottom: 20}}>
        <Text style={styles.heading}>Predict The Price of House</Text>
      </View>
      <ScrollView>
        <Text style={[styles.formText,{fontSize:20}]}>Bedrooms</Text>
        <TextInput
          onChangeText={setBedrooms}
          placeholder="Enter Number Of Bed rooms"
          style={styles.inputs}
          placeholderTextColor={'black'}
        />

        <Text style={[styles.formText,{fontSize:20}]}>Property Size</Text>
        <View style={styles.contentView}>
          <Text
            style={[styles.formText, {color: '#1ebf94', fontWeight: '500'}]}>
            Up to {propertySize}sqft
          </Text>

          <Slider
            value={propertySize}
            onValueChange={setPropertySize}
            maximumValue={6000}
            minimumValue={1000}
            step={1}
            allowTouchTrack
            trackStyle={{height: 5, backgroundColor: '#1ebf94'}}
            thumbStyle={{
              height: 20,
              width: 10,
              backgroundColor: '#1ebf94',
            }}
            thumbProps={{
              children: (
                <Icon
                  name="circle"
                  type="font-awesome"
                  size={10}
                  reverse
                  containerStyle={{bottom: 10, right: 10}}
                  color={'#1ebf94'}
                />
              ),
            }}
          />
        </View>

        {/*  */}
        <View style={styles.row}>
  <View style={styles.checkBoxRow}>
    <CheckBox
      isChecked={resale === 'Yes'}
      onClick={() => setResale(resale === 'Yes' ? 'No' : 'Yes')}
    />
    <Text style={[styles.formText, { fontWeight: '500' }]}>Resale</Text>
  </View>
  <View style={styles.checkBoxRow}>
    <CheckBox
      isChecked={maintenanceStaff === 'Yes'}
      onClick={() =>
        setMaintenanceStaff(
          maintenanceStaff === 'Yes' ? 'No' : 'Yes'
        )
      }
    />
    <Text style={[styles.formText, { fontWeight: '500'}]}>
      Maintenance Staff
    </Text>
  </View>
</View>

{/* Repeat this structure for other checkboxes */}


        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={gymnasium === 'Yes'}
      onClick={() => setGymnasium(gymnasium === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Gymnasium
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={swimmingPool === 'Yes'}
      onClick={() => setSwimmingPool(swimmingPool === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Swimming Pool
            </Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={landscapedGardens === 'Yes'}
      onClick={() => setLandscapedGardens(landscapedGardens === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              LandscapedGardens
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={joggingTrack === 'Yes'}
      onClick={() => setJoggingTrack(joggingTrack === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Jogging Track
            </Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={rainWaterHarvesting === 'Yes'}
      onClick={() => setRainWaterHarvesting(rainWaterHarvesting === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Rain Water Harvesting
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={indoorGames === 'Yes'}
      onClick={() => setIndoorGames(indoorGames === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Indoor Games
            </Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={shoppingmall === 'Yes'}
      onClick={() => setShoppingMall(shoppingmall === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Shopping Mall
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={interCom === 'Yes'}
      onClick={() => SetInterCom(interCom === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>InterCom</Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={sportsFacility === 'Yes'}
      onClick={() => setSportsFacility(sportsFacility === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Sports Facility
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={atm=== 'Yes'}
      onClick={() => setAtm(atm === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>ATM</Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={clubHouse === 'Yes'}
      onClick={() => setClubHouse(clubHouse === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Club House
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={school === 'Yes'}
      onClick={() => setSchool(school === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>School</Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={security === 'Yes'}
      onClick={() => SetSecurity(security === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              24X7 Security
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
            <CheckBox
      isChecked={powerBackup === 'Yes'}
      onClick={() => setPowerBackup(powerBackup === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Power Backup
            </Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={carparking === 'Yes'}
      onClick={() => setCarParking(carparking === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Car Parking
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={staffQuarter === 'Yes'}
      onClick={() => setStaffQuarter(staffQuarter === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Staff Quarter
            </Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={cafteria === 'Yes'}
      onClick={() => setCafteria(cafteria === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Cafeteria
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={multiPurposeRoom === 'Yes'}
      onClick={() => setMultiPurposeRoom(multiPurposeRoom === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Multi Purpose Room
            </Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={hospital === 'Yes'}
      onClick={() => setHospital(hospital === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>Hospital</Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={washingMachine === 'Yes'}
      onClick={() => setWashingMachine(washingMachine === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Washing Machine
            </Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={gasConnection === 'Yes'}
      onClick={() => setGasConnection(washingMachine=== 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Gas Connection
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={AC === 'Yes'}
      onClick={() => setAC(AC === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>AC</Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={wifi === 'Yes'}
      onClick={() => setWifi(wifi === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>Wifi</Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={childrenArea === 'Yes'}
      onClick={() => setChildrenArea(childrenArea === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Chilren's play Area
            </Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={lift === 'Yes'}
      onClick={() => setLift(lift === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Lift Available
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={bed=== 'Yes'}
      onClick={() => setBed(bed === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>Bed</Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={microWaveOwen === 'Yes'}
      onClick={() => setMIcroWaveOwen(microWaveOwen === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>
              Micro Wave owen
            </Text>
          </View>
          <View style={styles.checkBoxRow}>
            <CheckBox
      isChecked={TV === 'Yes'}
      onClick={() => setTV(TV=== 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>TV</Text>
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={sofa === 'Yes'}
      onClick={() => setSofa(sofa === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>Sofa</Text>
          </View>
          <View style={styles.checkBoxRow}>
          <CheckBox
      isChecked={wardrobe === 'Yes'}
      onClick={() => setWardrobe(wardrobe === 'Yes' ? 'No' : 'Yes')}
    />
            <Text style={[styles.formText, {fontWeight: '500'}]}>wardrobe</Text>
          </View>
        </View>
        {/*  */}

        {/*  */}

        <TouchableOpacity style={styles.btnsubmit} onPress={handleSubmit}>
          <Text style={[styles.btnText, {fontSize: 18, color: 'white'}]}>
            Predict Price
          </Text>
        </TouchableOpacity>
        {/*  */}
      </ScrollView>
    </View>
  );
};
const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginBottom:20
    // padding: 10,
    // backgroundColor:'white'
  },
  heading: {
    padding: 10,
    color: 'black',
    fontSize: 25,
    textAlign: 'center',
    fontWeight: 'bold',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    // marginBottom: 10,
    justifyContent:'flex-start',
    padding:10
  },
  checkBoxRow:{ 
    flexDirection: 'row',
     alignItems: 'center' ,
     width:200
  },

  head: {
    fontWeight: '500',
    fontSize: 28,
    color: 'black',
    textAlign: 'center',
  },
  formText: {
    fontWeight: '500',
    fontSize: 17,
    color: 'black',
    marginHorizontal: 5,
    // marginLeft:-80
  },
  btnGroup: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  btn: {
    flex: 1,
    borderWidth: 0.25,
    borderColor: '#6B7280',
    borderRadius: 5,
    marginRight: 10,
  },
  btnText: {
    textAlign: 'center',
    paddingVertical: 16,
    fontSize: 15,
    color: 'black',
  },
  contentView: {
    width: '97%',
    height: 100,
    justifyContent: 'center',
    alignItems: 'stretch',
    backgroundColor: 'white',
    padding: 10,
    borderRadius: 10,
  },
  btnsubmit: {
    backgroundColor: '#ff8717',
    height: 55,
    width: '95%',
    marginTop: 30,
    borderRadius: 30,
    alignSelf: 'center',
  },
  inputs: {
    borderWidth: 1,
    width: '95%',
    alignSelf: 'center',
    borderRadius: 10,
    marginBottom: 20,
    marginTop:10
  },
});
export default Home;
