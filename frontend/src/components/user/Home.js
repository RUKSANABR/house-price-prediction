import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
} from 'react-native';

import React, { useState } from 'react';
import { useNavigation } from '@react-navigation/native';
import { Picker } from '@react-native-picker/picker';
const Home = ({ route }) => {
  const navigation = useNavigation();
  const { userD } = route.params;
  const [msZoning, setMsZoning] = useState('RL'); // Default value for MSZoning
  const [lotArea, setLotArea] = useState(8000); // You can set an empty string or any other default value
  const [street, setStreet] = useState('Pave'); // Default value for Street
  const [conditional1, setConditional1] = useState('Norm'); // Default value for Condition1
  const [houseStyle, setHouseStyle] = useState('2Story'); // Default value for HouseStyle
  const [overallCond, setOverallCond] = useState(5); // Default value for OverallCond
  const [roofStyle, setRoofStyle] = useState('Gable'); // Default value for RoofStyle
  const [roofMat1, setRoofMat1] = useState('CompShg'); // You can set an empty string or any other default value
  const [foundation, setFoundation] = useState('PConc'); // Default value for Foundation
  const [bsmtQual, setBsmtQual] = useState('Gd'); // Default value for BsmtQual
  const [heating, setHeating] = useState('GasA'); // Default value for Heating
  const [heatingQC, setHeatingQC] = useState('Ex'); // Default value for HeatingQC
  const [centralAir, setCentralAir] = useState('Y'); // Default value for CentralAir
  const [electrical, setElectrical] = useState('SBrkr'); // Default value for Electrical
  const [fullBath, setFullBath] = useState(2); // You can set an empty string or any other default value
  const [kitchenQual, setKitchenQual] = useState('Gd'); // Default value for KitchenQual
  const [totRmsAbvGrd, setTotRmsAbvGrd] = useState(8); // You can set an empty string or any other default value
  const [functional, setFunctional] = useState('Typ'); // Default value for Functional
  const [garageType, setGarageType] = useState('Attchd'); // Default value for GarageType
  const [priceProperty, setpriceProperty] = useState(50000); // Default value for GarageType
  const [price, setPrice] = useState(null);

  const date = new Date();


  const color = () => { };



  const handleSubmit = async () => {
    try {
      console.log('calling')
      const apiUrl = `http://10.0.2.2:5000/api?query=${msZoning}|${lotArea}|${street}|${conditional1}|${houseStyle}|${overallCond}|${roofStyle}|${roofMat1}|${foundation}|${bsmtQual}|${heating}|${heatingQC}|${centralAir}|${electrical}|${fullBath}|${kitchenQual}|${totRmsAbvGrd}|${functional}|${garageType}`;
      const response = await fetch(apiUrl);
      const data = await response.json();
      const apiPrice = data['price'];
      // console.log(apiPrice);
      // setpriceProperty(apiPrice);
      // setPrice(apiPrice);
      // console.log("value set");
      // console.log(priceProperty);
      // console.log(apiPrice);

      const decimal = Math.round(apiPrice * 100) / 100
      console.log(apiPrice);
      setpriceProperty(decimal); // Update the state
      console.log("value set");
      console.log(apiPrice);


      //
      const propertyData = {
        price: decimal,
        MSZoning: msZoning,
        LotArea: lotArea,
        Street: street,
        Condition1: conditional1,
        HouseStyle: houseStyle,
        HouseStyle: overallCond,
        RoofStyle: roofStyle,
        RoofMatl: roofMat1,
        Foundation: foundation,
        BsmtQual: bsmtQual,
        Heating: heating,
        HeatingQC: heatingQC,
        CentralAir: centralAir,
        Electrical: electrical,
        FullBath: fullBath,
        KitchenQual: kitchenQual,
        TotRmsAbvGrd: totRmsAbvGrd,
        Functional: functional,
        GarageType: garageType,
        // location:userD.place,
        userName: userD.name,
        phone: userD.phone,
        userId: userD.email,
        status: 'Available',
        Date: date.toISOString(),
      };
      console.log(propertyData);

      navigation.navigate('Property', { userD, propertyData });

    } catch (error) {
      console.error('Error fetching data:', error);
    }


  };
  height: 40;
  return (
    <View style={styles.container}>
      <View style={{ backgroundColor: 'white', marginBottom: 20 }}>
        <Text style={styles.heading}>Predict The Price of House</Text>
      </View>
      <ScrollView>
        {/*  */}
        <Text style={styles.formText}>MSZoning</Text>
        <Picker
          selectedValue={msZoning}
          onValueChange={(itemValue) => setMsZoning(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Agriculture" value="A" />
          <Picker.Item label="Commercial" value="C" />
          <Picker.Item label="Floating Village Residential" value="FV" />
          <Picker.Item label="Industrial" value="I" />
          <Picker.Item label="Residential High Density" value="RH" />
          <Picker.Item label="Residential Low Density" value="RL" />
          <Picker.Item label="Residential Low Density Park" value="RP" />
          <Picker.Item label="Residential Medium Density" value="RM" />
        </Picker>
        {/*  */}
        <Text style={styles.formText}>LotArea</Text>
        <TextInput
          placeholder="Enter in Squre feet"
          style={styles.inputs} placeholderTextColor="black"
          onChangeText={setLotArea}></TextInput>
        {/*  */}
        <Text style={styles.formText}>Street</Text>
        <Picker
          selectedValue={street}
          onValueChange={(itemValue) => setStreet(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Gravel" value="Grvl" />
          <Picker.Item label="Paved" value="Pave" />
        </Picker>
        {/*  */}

        {/*  */}
        <Text style={styles.formText}>Condition1: Proximity to various conditions</Text>
        <Picker
          selectedValue={conditional1}
          onValueChange={(itemValue) => setConditional1(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Adjacent to arterial street" value="Artery" />
          <Picker.Item label="Adjacent to feeder street" value="Feedr" />
          <Picker.Item label="Normal" value="Norm" />
          <Picker.Item label="Within 200' of North-South Railroad" value="RRNn" />
          <Picker.Item label="Adjacent to North-South Railroad" value="RRAn" />
          <Picker.Item label="Near positive off-site feature--park, greenbelt, etc." value="PosN" />
          <Picker.Item label="Adjacent to postive off-site feature" value="PosA" />
          <Picker.Item label="Within 200' of East-West Railroad" value="RRNe" />
          <Picker.Item label="Adjacent to East-West Railroad" value="RRAe" />
        </Picker>
        {/*  */}

        <Text style={styles.formText}>HouseStyle</Text>
        <Picker
          selectedValue={houseStyle}
          onValueChange={(itemValue) => setHouseStyle(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="One story" value="1Story	" />
          <Picker.Item label="One and one-half story: 2nd level finished" value="1.5Fin" />
          <Picker.Item label="One and one-half story: 2nd level unfinished" value="1.5Unf" />
          <Picker.Item label="Two story" value="2Story" />
          <Picker.Item label="Two and one-half story: 2nd level finished" value="2.5Fin" />
          <Picker.Item label="Two and one-half story: 2nd level unfinished" value="2.5Unf" />
          <Picker.Item label="Split Foyer" value="SFoyer" />
          <Picker.Item label="Split Level" value="SLvl" />
        </Picker>
        {/*  */}
        <Text style={styles.formText}>OverallCond: Rates the overall condition of the house</Text>
        <Picker
          selectedValue={overallCond}
          onValueChange={(itemValue) => setOverallCond(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Very Excellent" value="10" />
          <Picker.Item label="Excellent" value="9" />
          <Picker.Item label="Very Good" value="8" />
          <Picker.Item label="Good" value="7" />
          <Picker.Item label="Above Average" value="6" />
          <Picker.Item label="Average" value="5" />
          <Picker.Item label="Below Average" value="4" />
          <Picker.Item label="Fair" value="3" />
          <Picker.Item label="Poor" value="2" />
          <Picker.Item label="Very Poor" value="1" />
        </Picker>
        {/*  */}
        {/*  */}

        <Text style={styles.formText}>RoofStyle</Text>
        <Picker
          selectedValue={roofStyle}
          onValueChange={(itemValue) => setRoofStyle(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Flat" value="1Story	" />
          <Picker.Item label="Gable" value="Gable" />
          <Picker.Item label="Gabrel (Barn)" value="Gambrel" />
          <Picker.Item label="Hip" value="Hip" />
          <Picker.Item label="Mansard" value="Mansard" />
          <Picker.Item label="Shed" value="Shed" />

        </Picker>
        {/*  */}

        <Text style={styles.formText}>Roof Material</Text>
        <Picker
          selectedValue={roofMat1}
          onValueChange={(itemValue) => setRoofMat1(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Clay or Tile" value="ClyTile" />
          <Picker.Item label="Standard (Composite) Shingle" value="CompShg" />
          <Picker.Item label="Membrane" value="Membrane" />
          <Picker.Item label="Metal" value="Metal" />
          <Picker.Item label="Gravel & Tar" value="Tar&Grv" />
          <Picker.Item label="Wood Shakes" value="WdShake" />
          <Picker.Item label="Wood Shingles" value="WdShngl" />
        </Picker>

        {/*  */}

        <Text style={styles.formText}>Foundation</Text>
        <Picker
          selectedValue={foundation}
          onValueChange={(itemValue) => setFoundation(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Brick & Tile" value="BrkTil" />
          <Picker.Item label="Cinder Block" value="CBlock" />
          <Picker.Item label="Poured Contrete" value="PConc" />
          <Picker.Item label="Slab" value="Slab" />
          <Picker.Item label="Stone" value="Stone" />
          <Picker.Item label="Wood" value="Wood" />
        </Picker>
        {/*  */}

        <Text style={styles.formText}>BsmtQual</Text>
        <Picker
          selectedValue={bsmtQual}
          onValueChange={(itemValue) => setBsmtQual(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Excellent (100+ inches)" value="Ex" />
          <Picker.Item label="Good (90-99 inches)" value="Gd" />
          <Picker.Item label="Typical (80-89 inches)" value="Ta" />
          <Picker.Item label="Fair (70-79 inches)" value="Ta" />
          <Picker.Item label="Poor (<70 inches)" value="Po" />
          <Picker.Item label="No Basement" value="NA" />
        </Picker>

        {/*  */}

        <Text style={styles.formText}>Heating</Text>
        <Picker
          selectedValue={heating}
          onValueChange={(itemValue) => setHeating(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Floor Furnace" value="Floor" />
          <Picker.Item label="Gas forced warm air furnace" value="GasA" />
          <Picker.Item label="Gas hot water or steam heat" value="GasW" />
          <Picker.Item label="Gravity furnace" value="Grav" />
          <Picker.Item label="Hot water or steam heat other than gas" value="OthW" />
          <Picker.Item label="Wall furnace" value="Wall" />
        </Picker>

        {/*  */}

        <Text style={styles.formText}>HeatingQC</Text>
        <Picker
          selectedValue={heatingQC}
          onValueChange={(itemValue) => setHeatingQC(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Excellent" value="Ex" />
          <Picker.Item label="Good" value="Gd" />
          <Picker.Item label="Average/Typical" value="TA" />
          <Picker.Item label="Fair" value="Fa" />
          <Picker.Item label="Poor" value="Po" />
        </Picker>

        {/*  */}

        <Text style={styles.formText}>CentralAir</Text>
        <Picker
          selectedValue={centralAir}
          onValueChange={(itemValue) => setCentralAir(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="No" value="N" />
          <Picker.Item label="Yes" value="Y" />
        </Picker>
        {/*  */}

        <Text style={styles.formText}>Electrical</Text>
        <Picker
          selectedValue={electrical}
          onValueChange={(itemValue) => setElectrical(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Standard Circuit Breakers & Romex" value="SBrkr" />
          <Picker.Item label="Fuse Box over 60 AMP and all Romex wiring (Average)" value="FuseA" />
          <Picker.Item label="60 AMP Fuse Box and mostly Romex wiring (Fair)" value="FuseF" />
          <Picker.Item label="60 AMP Fuse Box and mostly knob & tube wiring (poor)" value="FuseP" />
          <Picker.Item label="Mixed" value="Mix" />
        </Picker>
        {/* ? */}

        <Text style={styles.formText}>FullBath</Text>
        <TextInput
          placeholder=" Total rooms above grade (does not include bathrooms)"
          style={styles.inputs} placeholderTextColor="black"
          onChangeText={setFullBath}></TextInput>
        {/*  */}


        <Text style={styles.formText}>KitchenQual</Text>
        <Picker
          selectedValue={kitchenQual}
          onValueChange={(itemValue) => setKitchenQual(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Excellent" value="Ex" />
          <Picker.Item label="Good" value="Gd" />
          <Picker.Item label="Average/Typical" value="TA" />
          <Picker.Item label="Fair" value="Fa" />
          <Picker.Item label="Poor" value="Po" />
        </Picker>
        {/*  */}
        <Text style={styles.formText}>Total Rooms Abve Grade</Text>
        <TextInput
          placeholder=" Total rooms above grade (does not include bathrooms)"
          style={styles.inputs} placeholderTextColor="black"
          onChangeText={setTotRmsAbvGrd}></TextInput>
        {/*  */}

        <Text style={styles.formText}>Home functionality</Text>
        <Picker
          selectedValue={functional}
          onValueChange={(itemValue) => setFunctional(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="Typical Functionality" value="Typ" />
          <Picker.Item label="Minor Deductions 1" value="Min1" />
          <Picker.Item label="Minor Deductions 2" value="Min2" />
          <Picker.Item label="Moderate Deductions" value="Mod" />
          <Picker.Item label="Major Deductions 1" value="Maj1" />
          <Picker.Item label="Major Deductions 2" value="Maj2" />
          <Picker.Item label="Severely Damaged" value="Sev" />
          <Picker.Item label="Poor" value="Sal" />
        </Picker>
        {/*  */}
        <Text style={styles.formText}>GarageType: Garage location</Text>
        <Picker
          selectedValue={garageType}
          onValueChange={(itemValue) => setGarageType(itemValue)}
          style={styles.inputs}
        >
          <Picker.Item label="More than one type of garage" value="2Types" />
          <Picker.Item label="Attached to home" value="Attchd" />
          <Picker.Item label="Basement Garage" value="Basment" />
          <Picker.Item label="Built-In (Garage part of house - typically has room above garage)" value="BuiltIn" />
          <Picker.Item label="Car Port" value="CarPort" />
          <Picker.Item label="Detached from home" value="Detchd" />
          <Picker.Item label="No Garage" value="NA" />
        </Picker>



        <TouchableOpacity style={styles.btnsubmit} onPress={handleSubmit}>
          <Text style={[styles.btnText, { fontSize: 18, color: 'white' }]}>
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
    marginBottom: 20
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
    justifyContent: 'flex-start',
    padding: 10
  },
  checkBoxRow: {
    flexDirection: 'row',
    alignItems: 'center',
    width: 200
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
    marginHorizontal: 10,
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
    marginTop: 10,
    borderWidth: 1,
    padding: 5,
    borderColor: 'black'
  },
});
export default Home;
